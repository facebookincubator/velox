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

#include <limits>
#include "velox/buffer/Buffer.h"
#include "velox/common/base/VeloxException.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/prestosql/tests/CastBaseTest.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/TypeAliases.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

class CastExprTest : public functions::test::CastBaseTest {
 protected:
  CastExprTest() {
    exec::registerVectorFunction(
        "testing_dictionary",
        TestingDictionaryFunction::signatures(),
        std::make_unique<TestingDictionaryFunction>());
  }

  void setCastIntByTruncate(bool value) {
    queryCtx_->setConfigOverridesUnsafe({
        {core::QueryConfig::kCastIntByTruncate, std::to_string(value)},
    });
  }

  void setCastMatchStructByName(bool value) {
    queryCtx_->setConfigOverridesUnsafe({
        {core::QueryConfig::kCastMatchStructByName, std::to_string(value)},
    });
  }

  void setTimezone(const std::string& value) {
    queryCtx_->setConfigOverridesUnsafe({
        {core::QueryConfig::kSessionTimezone, value},
        {core::QueryConfig::kAdjustTimestampToTimezone, "true"},
    });
  }

  std::shared_ptr<core::ConstantTypedExpr> makeConstantNullExpr(TypeKind kind) {
    return std::make_shared<core::ConstantTypedExpr>(
        createType(kind, {}), variant(kind));
  }

  std::shared_ptr<core::CastTypedExpr> makeCastExpr(
      const core::TypedExprPtr& input,
      const TypePtr& toType,
      bool nullOnFailure) {
    std::vector<core::TypedExprPtr> inputs = {input};
    return std::make_shared<core::CastTypedExpr>(toType, inputs, nullOnFailure);
  }

  void testComplexCast(
      const std::string& fromExpression,
      const VectorPtr& data,
      const VectorPtr& expected,
      bool nullOnFailure = false) {
    auto rowVector = makeRowVector({data});
    auto rowType = asRowType(rowVector->type());
    auto castExpr = makeCastExpr(
        makeTypedExpr(fromExpression, rowType),
        expected->type(),
        nullOnFailure);
    exec::ExprSet exprSet({castExpr}, &execCtx_);

    const auto size = data->size();
    SelectivityVector rows(size);
    std::vector<VectorPtr> result(1);
    {
      exec::EvalCtx evalCtx(&execCtx_, &exprSet, rowVector.get());
      exprSet.eval(rows, evalCtx, result);

      assertEqualVectors(expected, result[0]);
    }

    // Test constant input.
    {
      // Use last element for constant.
      const auto index = size - 1;
      auto constantData = BaseVector::wrapInConstant(size, index, data);
      auto constantRow = makeRowVector({constantData});
      exec::EvalCtx evalCtx(&execCtx_, &exprSet, constantRow.get());
      exprSet.eval(rows, evalCtx, result);

      assertEqualVectors(
          BaseVector::wrapInConstant(size, index, expected), result[0]);
    }

    // Test dictionary input. It is not sufficient to wrap input in a dictionary
    // as it will be peeled off before calling "cast". Apply
    // testing_dictionary function to input to ensure that "cast" receives
    // dictionary input.
    {
      auto dictionaryCastExpr = makeCastExpr(
          makeTypedExpr(
              fmt::format("testing_dictionary({})", fromExpression), rowType),
          expected->type(),
          nullOnFailure);
      exec::ExprSet dictionaryExprSet({dictionaryCastExpr}, &execCtx_);
      exec::EvalCtx evalCtx(&execCtx_, &dictionaryExprSet, rowVector.get());
      dictionaryExprSet.eval(rows, evalCtx, result);

      auto indices = ::makeIndicesInReverse(size, pool());
      assertEqualVectors(wrapInDictionary(indices, size, expected), result[0]);
    }
  }

  /**
   * @tparam From Source type for cast
   * @tparam To Destination type for cast
   * @param typeString Cast type in string
   * @param input Input vector of type From
   * @param expectedResult Expected output vector of type To
   * @param inputNulls Input null indexes
   * @param expectedNulls Expected output null indexes
   */
  template <typename TFrom, typename TTo>
  void testCast(
      const std::string& typeString,
      std::vector<std::optional<TFrom>> input,
      std::vector<std::optional<TTo>> expectedResult,
      bool expectFailure = false,
      bool tryCast = false) {
    std::vector<TFrom> rawInput(input.size());
    for (auto index = 0; index < input.size(); index++) {
      if (input[index].has_value()) {
        rawInput[index] = input[index].value();
      }
    }
    // Create input vector using values and nulls
    auto inputVector = makeFlatVector(rawInput);

    for (auto index = 0; index < input.size(); index++) {
      if (!input[index].has_value()) {
        inputVector->setNull(index, true);
      }
    }
    auto rowVector = makeRowVector({inputVector});
    std::string castFunction = tryCast ? "try_cast" : "cast";
    if (expectFailure) {
      EXPECT_THROW(
          evaluate(
              fmt::format("{}(c0 as {})", castFunction, typeString), rowVector),
          VeloxUserError);
      return;
    }
    // run try cast and get the result vector
    auto result =
        evaluate(castFunction + "(c0 as " + typeString + ")", rowVector);
    auto expected = makeNullableFlatVector<TTo>(expectedResult);
    assertEqualVectors(expected, result);
  }

  template <typename T>
  void testIntToDecimalCasts() {
    // integer to short decimal
    auto input = makeFlatVector<T>({-3, -2, -1, 0, 55, 69, 72});
    testComplexCast(
        "c0",
        input,
        makeShortDecimalFlatVector(
            {-300, -200, -100, 0, 5'500, 6'900, 7'200}, DECIMAL(6, 2)));

    // integer to long decimal
    testComplexCast(
        "c0",
        input,
        makeLongDecimalFlatVector(
            {-30'000'000'000,
             -20'000'000'000,
             -10'000'000'000,
             0,
             550'000'000'000,
             690'000'000'000,
             720'000'000'000},
            DECIMAL(20, 10)));

    // Expected failures: allowed # of integers (precision - scale) in the
    // target
    VELOX_ASSERT_THROW(
        testComplexCast(
            "c0",
            makeFlatVector<T>(std::vector<T>{std::numeric_limits<T>::min()}),
            makeShortDecimalFlatVector({0}, DECIMAL(3, 1))),
        fmt::format(
            "Cannot cast {} '{}' to DECIMAL(3,1)",
            CppToType<T>::name,
            std::to_string(std::numeric_limits<T>::min())));
    VELOX_ASSERT_THROW(
        testComplexCast(
            "c0",
            makeFlatVector<T>(std::vector<T>{-100}),
            makeShortDecimalFlatVector({0}, DECIMAL(17, 16))),
        fmt::format(
            "Cannot cast {} '-100' to DECIMAL(17,16)", CppToType<T>::name));
    VELOX_ASSERT_THROW(
        testComplexCast(
            "c0",
            makeFlatVector<T>(std::vector<T>{100}),
            makeShortDecimalFlatVector({0}, DECIMAL(17, 16))),
        fmt::format(
            "Cannot cast {} '100' to DECIMAL(17,16)", CppToType<T>::name));
  }
};

TEST_F(CastExprTest, basics) {
  // Testing non-null or error cases
  const std::vector<std::optional<int32_t>> ii = {1, 2, 3, 100, -100};
  const std::vector<std::optional<double>> oo = {1.0, 2.0, 3.0, 100.0, -100.0};
  testCast<int32_t, double>(
      "double", {1, 2, 3, 100, -100}, {1.0, 2.0, 3.0, 100.0, -100.0});
  testCast<int32_t, std::string>(
      "string", {1, 2, 3, 100, -100}, {"1", "2", "3", "100", "-100"});
  testCast<std::string, int8_t>(
      "tinyint", {"1", "2", "3", "100", "-100"}, {1, 2, 3, 100, -100});
  testCast<double, int>(
      "int", {1.888, 2.5, 3.6, 100.44, -100.101}, {2, 3, 4, 100, -100});
  testCast<double, double>(
      "double",
      {1.888, 2.5, 3.6, 100.44, -100.101},
      {1.888, 2.5, 3.6, 100.44, -100.101});
  testCast<double, std::string>(
      "string",
      {1.888, 2.5, 3.6, 100.44, -100.101},
      {"1.888", "2.5", "3.6", "100.44", "-100.101"});
  testCast<double, double>(
      "double",
      {1.888, 2.5, 3.6, 100.44, -100.101},
      {1.888, 2.5, 3.6, 100.44, -100.101});
  testCast<double, float>(
      "float",
      {1.888, 2.5, 3.6, 100.44, -100.101},
      {1.888, 2.5, 3.6, 100.44, -100.101});
  testCast<bool, std::string>("string", {true, false}, {"true", "false"});
}

TEST_F(CastExprTest, stringToTimestamp) {
  testCast<std::string, Timestamp>(
      "timestamp",
      {
          "1970-01-01",
          "2000-01-01",
          "1970-01-01 00:00:00",
          "2000-01-01 12:21:56",
          "1970-01-01 00:00:00-02:00",
          std::nullopt,
      },
      {
          Timestamp(0, 0),
          Timestamp(946684800, 0),
          Timestamp(0, 0),
          Timestamp(946729316, 0),
          Timestamp(7200, 0),
          std::nullopt,
      });
}

TEST_F(CastExprTest, timestampToString) {
  testCast<Timestamp, std::string>(
      "string",
      {
          Timestamp(-946684800, 0),
          Timestamp(-7266, 0),
          Timestamp(0, 0),
          Timestamp(946684800, 0),
          Timestamp(9466848000, 0),
          Timestamp(94668480000, 0),
          Timestamp(946729316, 0),
          Timestamp(946729316, 123),
          Timestamp(946729316, 129900000),
          Timestamp(7266, 0),
          std::nullopt,
      },
      {
          "1940-01-02T00:00:00.000",
          "1969-12-31T21:58:54.000",
          "1970-01-01T00:00:00.000",
          "2000-01-01T00:00:00.000",
          "2269-12-29T00:00:00.000",
          "4969-12-04T00:00:00.000",
          "2000-01-01T12:21:56.000",
          "2000-01-01T12:21:56.000",
          "2000-01-01T12:21:56.129",
          "1970-01-01T02:01:06.000",
          std::nullopt,
      });
}

TEST_F(CastExprTest, dateToTimestamp) {
  testCast<Date, Timestamp>(
      "timestamp",
      {
          Date(0),
          Date(10957),
          Date(14557),
          std::nullopt,
      },
      {
          Timestamp(0, 0),
          Timestamp(946684800, 0),
          Timestamp(1257724800, 0),
          std::nullopt,
      });
}

TEST_F(CastExprTest, timestampToDate) {
  testCast<Timestamp, Date>(
      "date",
      {
          Timestamp(0, 0),
          Timestamp(946684800, 0),
          Timestamp(1257724800, 0),
          std::nullopt,
      },
      {
          Date(0),
          Date(10957),
          Date(14557),
          std::nullopt,
      });
}

TEST_F(CastExprTest, timestampInvalid) {
  testCast<int8_t, Timestamp>("timestamp", {12}, {Timestamp(0, 0)}, true);
  testCast<int16_t, Timestamp>("timestamp", {1234}, {Timestamp(0, 0)}, true);
  testCast<int32_t, Timestamp>("timestamp", {1234}, {Timestamp(0, 0)}, true);
  testCast<int64_t, Timestamp>("timestamp", {1234}, {Timestamp(0, 0)}, true);

  testCast<float, Timestamp>("timestamp", {12.99}, {Timestamp(0, 0)}, true);
  testCast<double, Timestamp>("timestamp", {12.99}, {Timestamp(0, 0)}, true);
}

TEST_F(CastExprTest, timestampAdjustToTimezone) {
  setTimezone("America/Los_Angeles");

  // Expect unix epochs to be converted to LA timezone (8h offset).
  testCast<std::string, Timestamp>(
      "timestamp",
      {
          "1970-01-01",
          "2000-01-01",
          "1969-12-31 16:00:00",
          "2000-01-01 12:21:56",
          "1970-01-01 00:00:00+14:00",
          std::nullopt,
          "2000-05-01", // daylight savings - 7h offset.
      },
      {
          Timestamp(28800, 0),
          Timestamp(946713600, 0),
          Timestamp(0, 0),
          Timestamp(946758116, 0),
          Timestamp(-21600, 0),
          std::nullopt,
          Timestamp(957164400, 0),
      });

  // Empty timezone is assumed to be GMT.
  setTimezone("");
  testCast<std::string, Timestamp>(
      "timestamp", {"1970-01-01"}, {Timestamp(0, 0)});
}

TEST_F(CastExprTest, timestampAdjustToTimezoneInvalid) {
  auto testFunc = [&]() {
    testCast<std::string, Timestamp>(
        "timestamp", {"1970-01-01"}, {Timestamp(1, 0)});
  };

  setTimezone("bla");
  EXPECT_THROW(testFunc(), std::runtime_error);
}

TEST_F(CastExprTest, date) {
  std::vector<std::optional<std::string>> input{
      "1970-01-01",
      "2020-01-01",
      "2135-11-09",
      "1969-12-27",
      "1812-04-15",
      "1920-01-02",
      std::nullopt,
  };
  std::vector<std::optional<Date>> result{
      Date(0),
      Date(18262),
      Date(60577),
      Date(-5),
      Date(-57604),
      Date(-18262),
      std::nullopt,
  };

  testCast<std::string, Date>("date", input, result);

  setCastIntByTruncate(true);
  testCast<std::string, Date>("date", input, result);
}

TEST_F(CastExprTest, invalidDate) {
  testCast<int8_t, Date>("date", {12}, {Date(0)}, true);
  testCast<int16_t, Date>("date", {1234}, {Date(0)}, true);
  testCast<int32_t, Date>("date", {1234}, {Date(0)}, true);
  testCast<int64_t, Date>("date", {1234}, {Date(0)}, true);

  testCast<float, Date>("date", {12.99}, {Date(0)}, true);
  testCast<double, Date>("date", {12.99}, {Date(0)}, true);

  // Parsing an ill-formated date.
  testCast<std::string, Date>("date", {"2012-Oct-23"}, {Date(0)}, true);
}

TEST_F(CastExprTest, truncateVsRound) {
  // Testing truncate vs round cast from double to int.
  setCastIntByTruncate(true);
  testCast<double, int>(
      "int", {1.888, 2.5, 3.6, 100.44, -100.101}, {1, 2, 3, 100, -100});
  testCast<double, int8_t>(
      "tinyint",
      {1,
       256,
       257,
       2147483646,
       2147483647,
       2147483648,
       -2147483646,
       -2147483647,
       -2147483648,
       -2147483649},
      {1, 0, 1, -2, -1, -1, 2, 1, 0, 0});

  setCastIntByTruncate(false);
  testCast<double, int>(
      "int", {1.888, 2.5, 3.6, 100.44, -100.101}, {2, 3, 4, 100, -100});

  testCast<int8_t, int32_t>("int", {111, 2, 3, 10, -10}, {111, 2, 3, 10, -10});

  setCastIntByTruncate(true);
  testCast<int32_t, int8_t>(
      "tinyint", {1111111, 2, 3, 1000, -100101}, {71, 2, 3, -24, -5});

  setCastIntByTruncate(false);
  EXPECT_THROW(
      (testCast<int32_t, int8_t>(
          "tinyint", {1111111, 2, 3, 1000, -100101}, {71, 2, 3, -24, -5})),
      VeloxUserError);
}

TEST_F(CastExprTest, nullInputs) {
  // Testing null inputs
  testCast<double, double>(
      "double",
      {std::nullopt, std::nullopt, 3.6, 100.44, std::nullopt},
      {std::nullopt, std::nullopt, 3.6, 100.44, std::nullopt});
  testCast<double, float>(
      "float",
      {std::nullopt, 2.5, 3.6, 100.44, std::nullopt},
      {std::nullopt, 2.5, 3.6, 100.44, std::nullopt});
  testCast<double, std::string>(
      "string",
      {1.888, std::nullopt, std::nullopt, std::nullopt, -100.101},
      {"1.888", std::nullopt, std::nullopt, std::nullopt, "-100.101"});
}

TEST_F(CastExprTest, errorHandling) {
  // Making sure error cases lead to null outputs
  testCast<std::string, int8_t>(
      "tinyint",
      {"1abc", "2", "3", "100", std::nullopt},
      {std::nullopt, 2, 3, 100, std::nullopt},
      false,
      true);

  setCastIntByTruncate(true);
  testCast<std::string, int8_t>(
      "tinyint",
      {"-",
       "-0",
       " @w 123",
       "123 ",
       "  122",
       "",
       "-12-3",
       "125.5",
       "1234",
       "-129",
       "127",
       "-128"},
      {std::nullopt,
       0,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       std::nullopt,
       127,
       -128},
      false,
      true);

  testCast<double, int>(
      "integer",
      {1e12, 2.5, 3.6, 100.44, -100.101},
      {std::numeric_limits<int32_t>::max(), 2, 3, 100, -100},
      false,
      true);

  setCastIntByTruncate(false);
  testCast<double, int>(
      "int", {1.888, 2.5, 3.6, 100.44, -100.101}, {2, 3, 4, 100, -100});

  testCast<std::string, int8_t>(
      "tinyint", {"1abc", "2", "3", "100", "-100"}, {1, 2, 3, 100, -100}, true);

  testCast<std::string, int8_t>(
      "tinyint", {"1", "2", "3", "100", "-100.5"}, {1, 2, 3, 100, -100}, true);
}

constexpr vector_size_t kVectorSize = 1'000;

TEST_F(CastExprTest, mapCast) {
  auto sizeAt = [](vector_size_t row) { return row % 5; };
  auto keyAt = [](vector_size_t row) { return row % 11; };
  auto valueAt = [](vector_size_t row) { return row % 13; };

  auto inputMap = makeMapVector<int64_t, int64_t>(
      kVectorSize, sizeAt, keyAt, valueAt, nullEvery(3));

  // Cast map<bigint, bigint> -> map<integer, double>.
  {
    auto expectedMap = makeMapVector<int32_t, double>(
        kVectorSize, sizeAt, keyAt, valueAt, nullEvery(3));

    testComplexCast("c0", inputMap, expectedMap);
  }

  // Cast map<bigint, bigint> -> map<bigint, varchar>.
  {
    auto valueAtString = [valueAt](vector_size_t row) {
      return StringView::makeInline(folly::to<std::string>(valueAt(row)));
    };

    auto expectedMap = makeMapVector<int64_t, StringView>(
        kVectorSize, sizeAt, keyAt, valueAtString, nullEvery(3));

    testComplexCast("c0", inputMap, expectedMap);
  }

  // Cast map<bigint, bigint> -> map<varchar, bigint>.
  {
    auto keyAtString = [&](vector_size_t row) {
      return StringView::makeInline(folly::to<std::string>(keyAt(row)));
    };

    auto expectedMap = makeMapVector<StringView, int64_t>(
        kVectorSize, sizeAt, keyAtString, valueAt, nullEvery(3));

    testComplexCast("c0", inputMap, expectedMap);
  }

  // null values
  {
    auto inputWithNullValues = makeMapVector<int64_t, int64_t>(
        kVectorSize, sizeAt, keyAt, valueAt, nullEvery(3), nullEvery(7));

    auto expectedMap = makeMapVector<int32_t, double>(
        kVectorSize, sizeAt, keyAt, valueAt, nullEvery(3), nullEvery(7));

    testComplexCast("c0", inputWithNullValues, expectedMap);
  }

  // Nulls in result keys are not allowed.
  {
    VELOX_ASSERT_THROW(
        testComplexCast(
            "c0",
            inputMap,
            makeMapVector<Timestamp, int64_t>(
                kVectorSize,
                sizeAt,
                [](auto /*row*/) { return Timestamp(); },
                valueAt,
                nullEvery(3),
                nullEvery(7)),
            false),
        "Failed to cast from BIGINT to TIMESTAMP: 0. Conversion to Timestamp is not supported");

    testComplexCast(
        "c0",
        inputMap,
        makeMapVector<Timestamp, int64_t>(
            kVectorSize,
            sizeAt,
            [](auto /*row*/) { return Timestamp(); },
            valueAt,
            [](auto row) { return row % 3 == 0 || row % 5 != 0; }),
        true);
  }

  // Make sure that the output of map cast has valid(copyable) data even for
  // non selected rows.
  {
    auto mapVector = vectorMaker_.mapVector<int32_t, int32_t>(
        kVectorSize,
        sizeAt,
        keyAt,
        /*valueAt*/ nullptr,
        /*isNullAt*/ nullptr,
        /*valueIsNullAt*/ nullEvery(1));

    SelectivityVector rows(5);
    rows.setValid(2, false);
    mapVector->setOffsetAndSize(2, 100, 100);
    std::vector<VectorPtr> results(1);

    auto rowVector = makeRowVector({mapVector});
    auto castExpr =
        makeTypedExpr("c0::map(bigint, bigint)", asRowType(rowVector->type()));
    exec::ExprSet exprSet({castExpr}, &execCtx_);

    exec::EvalCtx evalCtx(&execCtx_, &exprSet, rowVector.get());
    exprSet.eval(rows, evalCtx, results);
    auto mapResults = results[0]->as<MapVector>();
    auto keysSize = mapResults->mapKeys()->size();
    auto valuesSize = mapResults->mapValues()->size();

    for (int i = 0; i < mapResults->size(); i++) {
      auto start = mapResults->offsetAt(i);
      auto size = mapResults->sizeAt(i);
      if (size == 0) {
        continue;
      }
      VELOX_CHECK(start + size - 1 < keysSize);
      VELOX_CHECK(start + size - 1 < valuesSize);
    }
  }
}

TEST_F(CastExprTest, arrayCast) {
  auto sizeAt = [](vector_size_t /* row */) { return 7; };
  auto valueAt = [](vector_size_t /* row */, vector_size_t idx) {
    return 1 + idx;
  };
  auto arrayVector =
      makeArrayVector<double>(kVectorSize, sizeAt, valueAt, nullEvery(3));

  // Cast array<double> -> array<bigint>.
  {
    auto expected =
        makeArrayVector<int64_t>(kVectorSize, sizeAt, valueAt, nullEvery(3));
    testComplexCast("c0", arrayVector, expected);
  }

  // Cast array<double> -> array<varchar>.
  {
    auto valueAtString = [valueAt](vector_size_t row, vector_size_t idx) {
      return StringView::makeInline(folly::to<std::string>(valueAt(row, idx)));
    };
    auto expected = makeArrayVector<StringView>(
        kVectorSize, sizeAt, valueAtString, nullEvery(3));
    testComplexCast("c0", arrayVector, expected);
  }

  // Make sure that the output of array cast has valid(copyable) data even for
  // non selected rows.
  {
    // Array with all inner elements null.
    auto sizeAtLocal = [](vector_size_t /* row */) { return 5; };
    auto arrayVector = vectorMaker_.arrayVector<int32_t>(
        kVectorSize, sizeAtLocal, nullptr, nullptr, nullEvery(1));

    SelectivityVector rows(5);
    rows.setValid(2, false);
    arrayVector->setOffsetAndSize(2, 100, 10);
    std::vector<VectorPtr> results(1);

    auto rowVector = makeRowVector({arrayVector});
    auto castExpr =
        makeTypedExpr("cast (c0 as bigint[])", asRowType(rowVector->type()));
    exec::ExprSet exprSet({castExpr}, &execCtx_);

    exec::EvalCtx evalCtx(&execCtx_, &exprSet, rowVector.get());
    exprSet.eval(rows, evalCtx, results);
    auto arrayResults = results[0]->as<ArrayVector>();
    auto elementsSize = arrayResults->elements()->size();
    for (int i = 0; i < arrayResults->size(); i++) {
      auto start = arrayResults->offsetAt(i);
      auto size = arrayResults->sizeAt(i);
      if (size == 0) {
        continue;
      }
      VELOX_CHECK(start + size - 1 < elementsSize);
    }
  }
}

TEST_F(CastExprTest, rowCast) {
  auto valueAt = [](vector_size_t row) { return double(1 + row); };
  auto valueAtInt = [](vector_size_t row) { return int64_t(1 + row); };
  auto doubleVectorNullEvery3 =
      makeFlatVector<double>(kVectorSize, valueAt, nullEvery(3));
  auto intVectorNullEvery11 =
      makeFlatVector<int64_t>(kVectorSize, valueAtInt, nullEvery(11));
  auto doubleVectorNullEvery11 =
      makeFlatVector<double>(kVectorSize, valueAt, nullEvery(11));
  auto intVectorNullEvery3 =
      makeFlatVector<int64_t>(kVectorSize, valueAtInt, nullEvery(3));
  auto rowVector = makeRowVector(
      {intVectorNullEvery11, doubleVectorNullEvery3}, nullEvery(5));

  setCastMatchStructByName(false);
  // Position-based cast: ROW(c0: bigint, c1: double) -> ROW(c0: double, c1:
  // bigint)
  {
    auto expectedRowVector = makeRowVector(
        {doubleVectorNullEvery11, intVectorNullEvery3}, nullEvery(5));
    testComplexCast("c0", rowVector, expectedRowVector);
  }
  // Position-based cast: ROW(c0: bigint, c1: double) -> ROW(a: double, b:
  // bigint)
  {
    auto expectedRowVector = makeRowVector(
        {"a", "b"},
        {doubleVectorNullEvery11, intVectorNullEvery3},
        nullEvery(5));
    testComplexCast("c0", rowVector, expectedRowVector);
  }
  // Position-based cast: ROW(c0: bigint, c1: double) -> ROW(c0: double)
  {
    auto expectedRowVector =
        makeRowVector({doubleVectorNullEvery11}, nullEvery(5));
    testComplexCast("c0", rowVector, expectedRowVector);
  }

  // Name-based cast: ROW(c0: bigint, c1: double) -> ROW(c0: double) dropping
  // b
  setCastMatchStructByName(true);
  {
    auto intVectorNullAll = makeFlatVector<int64_t>(
        kVectorSize, valueAtInt, [](vector_size_t /* row */) { return true; });
    auto expectedRowVector = makeRowVector(
        {"c0", "b"}, {doubleVectorNullEvery11, intVectorNullAll}, nullEvery(5));
    testComplexCast("c0", rowVector, expectedRowVector);
  }
}

TEST_F(CastExprTest, nulls) {
  auto input =
      makeFlatVector<int32_t>(kVectorSize, [](auto row) { return row; });
  auto allNulls = makeFlatVector<int32_t>(
      kVectorSize, [](auto row) { return row; }, nullEvery(1));

  auto result = evaluate<FlatVector<int16_t>>(
      "cast(if(c0 % 2 = 0, c1, c0) as smallint)",
      makeRowVector({input, allNulls}));

  auto expectedResult = makeFlatVector<int16_t>(
      kVectorSize, [](auto row) { return row; }, nullEvery(2));
  assertEqualVectors(expectedResult, result);
}

TEST_F(CastExprTest, testNullOnFailure) {
  auto input =
      makeNullableFlatVector<std::string>({"1", "2", "", "3.4", std::nullopt});
  auto expected = makeNullableFlatVector<int32_t>(
      {1, 2, std::nullopt, std::nullopt, std::nullopt});

  // nullOnFailure is true, so we should return null instead of throwing.
  testComplexCast("c0", input, expected, true);

  // nullOnFailure is false, so we should throw.
  EXPECT_THROW(testComplexCast("c0", input, expected, false), VeloxUserError);
}

TEST_F(CastExprTest, toString) {
  auto input = std::make_shared<core::FieldAccessTypedExpr>(VARCHAR(), "a");
  exec::ExprSet exprSet(
      {makeCastExpr(input, BIGINT(), false),
       makeCastExpr(input, ARRAY(VARCHAR()), false)},
      &execCtx_);
  ASSERT_EQ("cast((a) as BIGINT)", exprSet.exprs()[0]->toString());
  ASSERT_EQ("cast((a) as ARRAY<VARCHAR>)", exprSet.exprs()[1]->toString());
}

TEST_F(CastExprTest, decimalToDecimal) {
  // short to short, scale up.
  auto shortFlat =
      makeShortDecimalFlatVector({-3, -2, -1, 0, 55, 69, 72}, DECIMAL(2, 2));
  testComplexCast(
      "c0",
      shortFlat,
      makeShortDecimalFlatVector(
          {-300, -200, -100, 0, 5'500, 6'900, 7'200}, DECIMAL(4, 4)));

  // short to short, scale down.
  testComplexCast(
      "c0",
      shortFlat,
      makeShortDecimalFlatVector({0, 0, 0, 0, 6, 7, 7}, DECIMAL(4, 1)));

  // long to short, scale up.
  auto longFlat =
      makeLongDecimalFlatVector({-201, -109, 0, 105, 208}, DECIMAL(20, 2));
  testComplexCast(
      "c0",
      longFlat,
      makeShortDecimalFlatVector(
          {-201'000, -109'000, 0, 105'000, 208'000}, DECIMAL(10, 5)));

  // long to short, scale down.
  testComplexCast(
      "c0",
      longFlat,
      makeShortDecimalFlatVector({-20, -11, 0, 11, 21}, DECIMAL(10, 1)));

  // long to long, scale up.
  testComplexCast(
      "c0",
      longFlat,
      makeLongDecimalFlatVector(
          {-20'100'000'000, -10'900'000'000, 0, 10'500'000'000, 20'800'000'000},
          DECIMAL(20, 10)));

  // long to long, scale down.
  testComplexCast(
      "c0",
      longFlat,
      makeLongDecimalFlatVector({-20, -11, 0, 11, 21}, DECIMAL(20, 1)));

  // short to long, scale up.
  testComplexCast(
      "c0",
      shortFlat,
      makeLongDecimalFlatVector(
          {-3'000'000'000,
           -2'000'000'000,
           -1'000'000'000,
           0,
           55'000'000'000,
           69'000'000'000,
           72'000'000'000},
          DECIMAL(20, 11)));

  // short to long, scale down.
  testComplexCast(
      "c0",
      makeShortDecimalFlatVector(
          {-20'500, -190, 12'345, 19'999}, DECIMAL(6, 4)),
      makeLongDecimalFlatVector({-21, 0, 12, 20}, DECIMAL(20, 1)));

  // NULLs and overflow.
  longFlat = makeNullableLongDecimalFlatVector(
      {-20'000, -1'000'000, 10'000, std::nullopt}, DECIMAL(20, 3));
  auto expectedShort = makeNullableShortDecimalFlatVector(
      {-200'000, std::nullopt, 100'000, std::nullopt}, DECIMAL(6, 4));

  // Throws exception if CAST fails.
  VELOX_ASSERT_THROW(
      testComplexCast("c0", longFlat, expectedShort),
      "Cannot cast DECIMAL '-1000.000' to DECIMAL(6,4)");

  // nullOnFailure is true.
  testComplexCast("c0", longFlat, expectedShort, true);

  // long to short, big numbers.
  testComplexCast(
      "c0",
      makeNullableLongDecimalFlatVector(
          {buildInt128(-2, 200),
           buildInt128(-1, 300),
           buildInt128(0, 400),
           buildInt128(1, 1),
           buildInt128(10, 100),
           std::nullopt},
          DECIMAL(23, 8)),
      makeNullableShortDecimalFlatVector(
          {-368934881474,
           -184467440737,
           0,
           184467440737,
           std::nullopt,
           std::nullopt},
          DECIMAL(12, 0)),
      true);

  // Overflow case.
  VELOX_ASSERT_THROW(
      testComplexCast(
          "c0",
          makeNullableLongDecimalFlatVector(
              {UnscaledLongDecimal::max().unscaledValue()}, DECIMAL(38, 0)),
          makeNullableLongDecimalFlatVector({0}, DECIMAL(38, 1))),
      "Cannot cast DECIMAL '99999999999999999999999999999999999999' to DECIMAL(38,1)");
  VELOX_ASSERT_THROW(
      testComplexCast(
          "c0",
          makeNullableLongDecimalFlatVector(
              {UnscaledLongDecimal::min().unscaledValue()}, DECIMAL(38, 0)),
          makeNullableLongDecimalFlatVector({0}, DECIMAL(38, 1))),
      "Cannot cast DECIMAL '-99999999999999999999999999999999999999' to DECIMAL(38,1)");
}

TEST_F(CastExprTest, integerToDecimal) {
  testIntToDecimalCasts<int8_t>();
  testIntToDecimalCasts<int16_t>();
  testIntToDecimalCasts<int32_t>();
  testIntToDecimalCasts<int64_t>();
}

TEST_F(CastExprTest, castInTry) {
  // Test try(cast(array(varchar) as array(bigint))) whose input vector is
  // wrapped in dictinary encoding. The row of ["2a"] should trigger an error
  // during casting and the try expression should turn this error into a null at
  // this row.
  auto input = makeRowVector({makeNullableArrayVector<StringView>(
      {{{"1"_sv}}, {{"2a"_sv}}, std::nullopt, std::nullopt})});
  auto expected = makeNullableArrayVector<int64_t>(
      {{{1}}, std::nullopt, std::nullopt, std::nullopt});

  evaluateAndVerifyCastInTryDictEncoding(
      ARRAY(VARCHAR()), ARRAY(BIGINT()), input, expected);

  // Test try(cast(map(varchar, bigint) as map(bigint, bigint))) where "3a"
  // should trigger an error at the first row.
  auto map = makeRowVector({makeMapVector<StringView, int64_t>(
      {{{"1", 2}, {"3a", 4}}, {{"5", 6}, {"7", 8}}})});
  auto mapExpected = makeNullableMapVector<int64_t, int64_t>(
      {std::nullopt, {{{5, 6}, {7, 8}}}});
  evaluateAndVerifyCastInTryDictEncoding(
      MAP(VARCHAR(), BIGINT()), MAP(BIGINT(), BIGINT()), map, mapExpected);

  // Test try(cast(array(varchar) as array(bigint))) where "2a" should trigger
  // an error at the first row.
  auto array =
      makeArrayVector<StringView>({{"1"_sv, "2a"_sv}, {"3"_sv, "4"_sv}});
  auto arrayExpected =
      vectorMaker_.arrayVectorNullable<int64_t>({std::nullopt, {{3, 4}}});
  evaluateAndVerifyCastInTryDictEncoding(
      ARRAY(VARCHAR()), ARRAY(BIGINT()), makeRowVector({array}), arrayExpected);

  arrayExpected = vectorMaker_.arrayVectorNullable<int64_t>(
      {std::nullopt, std::nullopt, std::nullopt});
  evaluateAndVerifyCastInTryDictEncoding(
      ARRAY(VARCHAR()),
      ARRAY(BIGINT()),
      makeRowVector({BaseVector::wrapInConstant(3, 0, array)}),
      arrayExpected);

  auto nested = makeRowVector({makeNullableNestedArrayVector<StringView>(
      {{{{{"1"_sv, "2"_sv}}, {{"3"_sv}}, {{"4a"_sv, "5"_sv}}}},
       {{{{"6"_sv, "7"_sv}}}}})});
  auto nestedExpected =
      makeNullableNestedArrayVector<int64_t>({std::nullopt, {{{{6, 7}}}}});
  evaluateAndVerifyCastInTryDictEncoding(
      ARRAY(ARRAY(VARCHAR())), ARRAY(ARRAY(BIGINT())), nested, nestedExpected);
}

TEST_F(CastExprTest, primitiveNullConstant) {
  // Evaluate cast(NULL::double as bigint).
  auto cast =
      makeCastExpr(makeConstantNullExpr(TypeKind::DOUBLE), BIGINT(), false);

  auto result = evaluate(
      cast, makeRowVector({makeFlatVector<int64_t>(std::vector<int64_t>{1})}));
  auto expectedResult = makeNullableFlatVector<int64_t>({std::nullopt});
  assertEqualVectors(expectedResult, result);

  // Evaluate cast(try_cast(NULL::varchar as double) as bigint).
  auto innerCast =
      makeCastExpr(makeConstantNullExpr(TypeKind::VARCHAR), DOUBLE(), true);
  auto outerCast = makeCastExpr(innerCast, BIGINT(), false);

  result = evaluate(outerCast, makeRowVector(ROW({}, {}), 1));
  assertEqualVectors(expectedResult, result);
}

TEST_F(CastExprTest, primitiveWithDictionaryIntroducedNulls) {
  exec::registerVectorFunction(
      "add_dict",
      TestingDictionaryFunction::signatures(),
      std::make_unique<TestingDictionaryFunction>(2));

  {
    auto data = makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = evaluate(
        "cast(add_dict(add_dict(c0)) as smallint)", makeRowVector({data}));
    auto expected = makeNullableFlatVector<int16_t>(
        {std::nullopt,
         std::nullopt,
         3,
         4,
         5,
         6,
         7,
         std::nullopt,
         std::nullopt});
    assertEqualVectors(expected, result);
  }

  {
    auto data = makeNullableFlatVector<int64_t>(
        {1,
         2,
         std::nullopt,
         std::nullopt,
         std::nullopt,
         std::nullopt,
         std::nullopt,
         8,
         9});
    auto result = evaluate(
        "cast(add_dict(add_dict(c0)) as varchar)", makeRowVector({data}));
    auto expected = makeNullConstant(TypeKind::VARCHAR, 9);
    assertEqualVectors(expected, result);
  }
}

TEST_F(CastExprTest, castAsCall) {
  // Invoking cast through a CallExpr instead of a CastExpr
  const std::vector<std::optional<int32_t>> inputValues = {1, 2, 3, 100, -100};
  const std::vector<std::optional<double>> outputValues = {
      1.0, 2.0, 3.0, 100.0, -100.0};

  auto input = makeRowVector({makeNullableFlatVector(inputValues)});
  core::TypedExprPtr inputField =
      std::make_shared<const core::FieldAccessTypedExpr>(INTEGER(), "c0");
  core::TypedExprPtr callExpr = std::make_shared<const core::CallTypedExpr>(
      DOUBLE(), std::vector<core::TypedExprPtr>{inputField}, "cast");

  auto result = evaluate(callExpr, input);
  auto expected = makeNullableFlatVector(outputValues);
  assertEqualVectors(expected, result);
}

namespace {
/// Wraps input in a constant encoding that repeats the first element and then
/// in dictionary that reverses the order of rows.
class TestingDictionaryOverConstFunction : public exec::VectorFunction {
 public:
  TestingDictionaryOverConstFunction() {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /*outputType*/,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    const auto size = rows.size();
    auto constant = BaseVector::wrapInConstant(size, 0, args[0]);

    auto indices = makeIndicesInReverse(size, context.pool());
    auto nulls = allocateNulls(size, context.pool());
    result =
        BaseVector::wrapInDictionary(nulls, indices, size, std::move(constant));
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // T -> T
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("T")
                .argumentType("T")
                .build()};
  }
};
} // namespace

TEST_F(CastExprTest, dictionaryOverConst) {
  // Verify that cast properly handles an input where the vector has a
  // dictionary layer wrapped over a constant layer.
  exec::registerVectorFunction(
      "dictionary_over_const",
      TestingDictionaryOverConstFunction::signatures(),
      std::make_unique<TestingDictionaryOverConstFunction>());

  auto data = makeFlatVector<int64_t>({1, 2, 3, 4, 5});
  auto result = evaluate(
      "cast(dictionary_over_const(c0) as smallint)", makeRowVector({data}));
  auto expected = makeNullableFlatVector<int16_t>({1, 1, 1, 1, 1});
  assertEqualVectors(expected, result);
}

namespace {
// Wrap input in a dictionary that point to subset of rows of the inner vector.
class TestingDictionaryToFewerRowsFunction : public exec::VectorFunction {
 public:
  TestingDictionaryToFewerRowsFunction() {}

  bool isDefaultNullBehavior() const override {
    return false;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /*outputType*/,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    const auto size = rows.size();
    auto indices = makeIndices(
        size, [](auto /*row*/) { return 0; }, context.pool());

    result = BaseVector::wrapInDictionary(nullptr, indices, size, args[0]);
  }

  static std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
    // T -> T
    return {exec::FunctionSignatureBuilder()
                .typeVariable("T")
                .returnType("T")
                .argumentType("T")
                .build()};
  }
};

} // namespace

TEST_F(CastExprTest, dictionaryEncodedNestedInput) {
  // Cast ARRAY<ROW<BIGINT>> to ARRAY<ROW<VARCHAR>> where the outermost ARRAY
  // layer and innermost BIGINT layer are dictionary-encoded. This test case
  // ensures that when casting the ROW<BIGINT> vector, the result ROW vector
  // would not be longer than the result VARCHAR vector. In the test below, the
  // ARRAY vector has 2 rows, each containing 3 elements. The ARRAY vector is
  // wrapped in a dictionary layer that only references its first row, hence
  // only the first 3 out of 6 rows are evaluated for the ROW and BIGINT vector.
  // The BIGINT vector is also dictionary-encoded, so CastExpr produces a result
  // VARCHAR vector of length 3. If the casting of the ROW vector produces a
  // result ROW<VARCHAR> vector of the length of all rows, i.e., 6, the
  // subsequent call to Expr::addNull() would throw due to the attempt of
  // accessing the element VARCHAR vector at indices corresonding to the
  // non-existent ROW at indices 3--5.
  exec::registerVectorFunction(
      "add_dict",
      TestingDictionaryToFewerRowsFunction::signatures(),
      std::make_unique<TestingDictionaryToFewerRowsFunction>());

  auto elements = makeFlatVector<int64_t>({1, 2, 3, 4, 5, 6});
  auto elementsInDict = BaseVector::wrapInDictionary(
      nullptr, makeIndices(6, [](auto row) { return row; }), 6, elements);
  auto row = makeRowVector({elementsInDict}, [](auto row) { return row == 2; });
  auto array = makeArrayVector({0, 3}, row);

  auto result = evaluate(
      "cast(add_dict(c0) as STRUCT(i varchar)[])", makeRowVector({array}));

  auto expectedElements = makeNullableFlatVector<StringView>(
      {"1"_sv, "2"_sv, "n"_sv, "1"_sv, "2"_sv, "n"_sv});
  auto expectedRow =
      makeRowVector({expectedElements}, [](auto row) { return row % 3 == 2; });
  auto expectedArray = makeArrayVector({0, 3}, expectedRow);
  assertEqualVectors(expectedArray, result);
}

TEST_F(CastExprTest, smallerNonNullRowsSizeThanRows) {
  // Evaluating Cast in Coalesce as the second argument triggers the copy of
  // Cast localResult to the result vector. The localResult vector is of the
  // size nonNullRows which can be smaller than rows. This test ensures that
  // Cast doesn't attempt to access values out-of-bound and hit errors.
  exec::registerVectorFunction(
      "add_dict_with_2_trailing_nulls",
      TestingDictionaryFunction::signatures(),
      std::make_unique<TestingDictionaryFunction>(2));

  auto data = makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3, 4}),
       makeNullableFlatVector<double>({std::nullopt, 6, 7, std::nullopt})});
  auto result = evaluate(
      "coalesce(c1, cast(add_dict_with_2_trailing_nulls(c0) as double))", data);
  auto expected = makeNullableFlatVector<double>({4, 6, 7, std::nullopt});
  assertEqualVectors(expected, result);
}
