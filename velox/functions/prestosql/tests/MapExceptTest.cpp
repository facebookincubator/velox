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
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/prestosql/tests/utils/FuzzerTestUtils.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/vector/FlatMapVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox::test;

namespace facebook::velox::functions {
namespace {

class MapExceptTest : public test::FunctionBaseTest {
 public:
  template <typename T>
  void testFloatNaNs() {
    static const auto kNaN = std::numeric_limits<T>::quiet_NaN();
    static const auto kSNaN = std::numeric_limits<T>::signaling_NaN();

    // Case 1: Non-constant search keys.
    auto data = makeRowVector(
        {makeMapVectorFromJson<T, int32_t>({
             "{1:10, NaN:20, 3:null, 4:40, 5:50, 6:60}",
             "{NaN:20}",
         }),
         makeArrayVector<T>({{1, kNaN, 5}, {kSNaN, 3}})});

    auto expected = makeMapVectorFromJson<T, int32_t>({
        "{3:null, 4:40, 6:60}",
        "{}",
    });
    auto result = evaluate("map_except(c0, c1)", data);
    assertEqualVectors(expected, result);

    // Case 2: Constant search keys.
    data = makeRowVector(
        {makeMapVectorFromJson<T, int32_t>({
             "{1:10, NaN:20, 3:null, 4:40, 5:50, 6:60}",
             "{NaN:20}",
         }),
         BaseVector::wrapInConstant(2, 0, makeArrayVector<T>({{1, kNaN, 5}}))});
    expected = makeMapVectorFromJson<T, int32_t>({
        "{3:null, 4:40, 6:60}",
        "{}",
    });
    result = evaluate("map_except(c0, c1)", data);
    assertEqualVectors(expected, result);
  }
};

TEST_F(MapExceptTest, bigintKey) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20, 3:null, 4:40, 5:50, 6:60}",
          "{1:10, 2:20, 4:40, 5:50}",
          "{}",
          "{2:20, 4:40, 6:60}",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[1, 3, 5]",
          "[1, 3, 5, 7]",
          "[3, 5]",
          "[1, 3]",
      }),
  });

  // Constant keys.
  auto result = evaluate("map_except(c0, array_constructor(1, 3, 5))", data);

  auto expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{2:20, 4:40, 6:60}",
      "{2:20, 4:40}",
      "{}",
      "{2:20, 4:40, 6:60}",
  });

  assertEqualVectors(expected, result);

  // Non-constant keys.
  result = evaluate("map_except(c0, c1)", data);
  assertEqualVectors(expected, result);

  // Empty list of keys. Expect all map entries returned.
  result = evaluate("map_except(c0, array_constructor()::bigint[])", data);

  expected = makeMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:null, 4:40, 5:50, 6:60}",
      "{1:10, 2:20, 4:40, 5:50}",
      "{}",
      "{2:20, 4:40, 6:60}",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapExceptTest, varcharKey) {
  auto data = makeRowVector({
      makeMapVectorFromJson<std::string, int32_t>({
          R"({"apple": 1, "banana": 2, "Cucurbitaceae": null, "date": 4, "eggplant": 5, "fig": 6})",
          R"({"banana": 2, "orange": 4})",
          R"({"banana": 2, "fig": 4, "date": 5})",
      }),
      makeArrayVectorFromJson<std::string>({
          R"(["apple", "Cucurbitaceae", "fig"])",
          R"(["apple", "Cucurbitaceae", "date", "eggplant"])",
          R"(["fig"])",
      }),
  });

  // Constant keys.
  auto result = evaluate(
      "map_except(c0, array_constructor('apple', 'some very looooong name', 'fig', 'Cucurbitaceae'))",
      data);

  auto expected = makeMapVectorFromJson<std::string, int32_t>({
      R"({"banana": 2, "date": 4, "eggplant": 5})",
      R"({"banana": 2, "orange": 4})",
      R"({"banana": 2, "date": 5})",
  });

  assertEqualVectors(expected, result);

  // Non-constant keys.
  result = evaluate("map_except(c0, c1)", data);
  assertEqualVectors(expected, result);

  // Empty list of keys. Expect all map entries returned.
  result = evaluate("map_except(c0, array_constructor()::varchar[])", data);

  expected = makeMapVectorFromJson<std::string, int32_t>({
      R"({"apple": 1, "banana": 2, "Cucurbitaceae": null, "date": 4, "eggplant": 5, "fig": 6})",
      R"({"banana": 2, "orange": 4})",
      R"({"banana": 2, "fig": 4, "date": 5})",
  });

  assertEqualVectors(expected, result);
}

TEST_F(MapExceptTest, floatNaNs) {
  testFloatNaNs<float>();
  testFloatNaNs<double>();
}

TEST_F(MapExceptTest, removeEveryKey) {
  auto data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({"{1:10, 2:20}", "{}", "{3:30}"}),
      makeArrayVectorFromJson<int64_t>({"[2, null, 1]", "[1]", "[4]"}),
  });
  auto expected =
      makeMapVectorFromJson<int64_t, int32_t>({"{}", "{}", "{3:30}"});
  assertEqualVectors(expected, evaluate("map_except(c0, c1)", data));

  data = makeRowVector({
      makeMapVectorFromJson<int64_t, int32_t>({"{1:10}", "{2:20}"}),
      makeArrayVectorFromJson<int64_t>({"[1]", "[2]"}),
  });
  expected = makeMapVectorFromJson<int64_t, int32_t>({"{}", "{}"});
  assertEqualVectors(expected, evaluate("map_except(c0, c1)", data));
}

TEST_F(MapExceptTest, complexValues) {
  auto data = makeRowVector({
      makeMapVector(
          {0, 3},
          makeFlatVector<int64_t>({1, 2, 3, 1, 2}),
          makeArrayVectorFromJson<int32_t>(
              {"[1, 2]", "null", "[3]", "[]", "[4, 5]"})),
      makeArrayVectorFromJson<int64_t>({"[2]", "[1, 3]"}),
  });
  auto expected = makeMapVector(
      {0, 2},
      makeFlatVector<int64_t>({1, 3, 2}),
      makeArrayVectorFromJson<int32_t>({"[1, 2]", "[3]", "[4, 5]"}));
  assertEqualVectors(expected, evaluate("map_except(c0, c1)", data));
}

TEST_F(MapExceptTest, wrappedMap) {
  auto maps = makeMapVectorFromJson<int64_t, int32_t>(
      {"{1:10, 2:20}", "{3:30}", "{1:11, 4:40}"});
  auto data = makeRowVector({
      wrapInDictionary(makeIndices({2, 0, 2, 1}), maps),
      makeArrayVectorFromJson<int64_t>({"[1]", "[2]", "[4]", "[5]"}),
  });
  auto expected = makeMapVectorFromJson<int64_t, int32_t>(
      {"{4:40}", "{1:10}", "{1:11}", "{3:30}"});
  assertEqualVectors(expected, evaluate("map_except(c0, c1)", data));
}

TEST_F(MapExceptTest, tooManyEntries) {
  // Every row reads the same 220,000-entry map, for 2.2B entries in total.
  constexpr vector_size_t kNumRows = 10'000;
  auto map = makeMapVector<int64_t, int64_t>(
      1,
      [](auto /*row*/) { return 220'000; },
      [](auto i) { return i; },
      [](auto i) { return i; });
  auto data = makeRowVector({
      wrapInDictionary(makeIndices(kNumRows, [](auto) { return 0; }), map),
      makeArrayVector<int64_t>(
          kNumRows, [](auto) { return 1; }, [](auto i) { return i; }),
  });
  VELOX_ASSERT_USER_THROW(
      evaluate("map_except(c0, c1)", data), "map_except input too large");
}

TEST_F(MapExceptTest, otherKeyTypes) {
  auto data = makeRowVector({
      makeMapVector<bool, int32_t>({{{true, 1}, {false, 2}}, {{false, 3}}}),
      makeArrayVector<bool>({{true}, {false}}),
  });
  auto expected = makeMapVector<bool, int32_t>({{{false, 2}}, {}});
  assertEqualVectors(expected, evaluate("map_except(c0, c1)", data));

  data = makeRowVector({
      makeMapVector<int32_t, int32_t>(
          {{{18'000, 1}, {18'001, 2}}}, MAP(DATE(), INTEGER())),
      makeArrayVector<int32_t>({{18'001}}, DATE()),
  });
  expected =
      makeMapVector<int32_t, int32_t>({{{18'000, 1}}}, MAP(DATE(), INTEGER()));
  assertEqualVectors(expected, evaluate("map_except(c0, c1)", data));

  data = makeRowVector({
      makeMapVector<Timestamp, int32_t>(
          {{{Timestamp(1, 0), 1}, {Timestamp(2, 0), 2}}}),
      makeArrayVector<Timestamp>({{Timestamp(1, 0)}}),
  });
  expected = makeMapVector<Timestamp, int32_t>({{{Timestamp(2, 0), 2}}});
  assertEqualVectors(expected, evaluate("map_except(c0, c1)", data));
}

TEST_F(MapExceptTest, arrayKey) {
  // NaNs match each other, and so do nulls nested in a key.
  auto data = makeRowVector({
      makeMapVector(
          {0, 3, 5},
          makeArrayVectorFromJson<double>({
              "[1, 2]",
              "[3]",
              "[]",
              "[NaN, 1]",
              "[null, 2]",
              "[1, 2]",
          }),
          makeFlatVector<int32_t>({10, 20, 30, 40, 50, 60})),
      makeArrayVector(
          {0, 2, 4},
          makeArrayVectorFromJson<double>({
              "[1, 2]",
              "[]",
              "[NaN, 1]",
              "[null, 2]",
              "[2, 1]",
          })),
  });
  auto expected = makeMapVector(
      {0, 1, 1},
      makeArrayVectorFromJson<double>({"[3]", "[1, 2]"}),
      makeFlatVector<int32_t>({20, 60}));
  assertEqualVectors(expected, evaluate("map_except(c0, c1)", data));

  // Constant keys, with a null element.
  data = makeRowVector({
      data->childAt(0),
      BaseVector::wrapInConstant(
          3,
          0,
          makeArrayVector(
              {0},
              makeArrayVectorFromJson<double>({"[1, 2]", "null", "[NaN, 1]"}))),
  });
  expected = makeMapVector(
      {0, 2, 3},
      makeArrayVectorFromJson<double>({"[3]", "[]", "[null, 2]"}),
      makeFlatVector<int32_t>({20, 30, 50}));
  assertEqualVectors(expected, evaluate("map_except(c0, c1)", data));
}

TEST_F(MapExceptTest, rowKey) {
  auto keyType = ROW({INTEGER(), VARCHAR()});
  auto data = makeRowVector({
      makeMapVector(
          {0, 3},
          makeRowVector(
              {makeFlatVector<int32_t>({1, 1, 2, 1}),
               makeFlatVector<std::string>({"a", "b", "a", "a"})}),
          makeFlatVector<int32_t>({10, 20, 30, 40})),
      makeArrayVector(
          {0, 2},
          makeRowVector(
              {makeFlatVector<int32_t>({1, 2, 1}),
               makeFlatVector<std::string>({"b", "a", "c"})})),
  });
  auto expected = makeMapVector(
      {0, 1},
      makeRowVector(
          {makeFlatVector<int32_t>({1, 1}),
           makeFlatVector<std::string>({"a", "a"})}),
      makeFlatVector<int32_t>({10, 40}));
  assertEqualVectors(expected, evaluate("map_except(c0, c1)", data));
}

TEST_F(MapExceptTest, longDecimalKey) {
  auto data = makeRowVector({
      makeMapVector(
          {0, 2},
          makeFlatVector<int128_t>({100, 200, 300}, DECIMAL(20, 2)),
          makeFlatVector<int32_t>({1, 2, 3})),
      makeArrayVector(
          {0, 1}, makeFlatVector<int128_t>({200, 400}, DECIMAL(20, 2))),
  });
  auto expected = makeMapVector(
      {0, 1},
      makeFlatVector<int128_t>({100, 300}, DECIMAL(20, 2)),
      makeFlatVector<int32_t>({1, 3}));
  assertEqualVectors(expected, evaluate("map_except(c0, c1)", data));
}

TEST_F(MapExceptTest, timestampWithTimeZoneKey) {
  // Keys are compared by instant, regardless of time zone.
  auto data = makeRowVector({
      makeMapVector(
          {0, 3},
          makeFlatVector<int64_t>(
              {pack(1, 1), pack(2, 2), pack(3, 3), pack(1, 4), pack(2, 5)},
              TIMESTAMP_WITH_TIME_ZONE()),
          makeFlatVector<int32_t>({10, 20, 30, 40, 50})),
      makeArrayVector(
          {0, 2},
          makeFlatVector<int64_t>(
              {pack(1, 7), pack(3, 8), pack(2, 9)},
              TIMESTAMP_WITH_TIME_ZONE())),
  });
  auto expected = makeMapVector(
      {0, 1},
      makeFlatVector<int64_t>(
          {pack(2, 2), pack(1, 4)}, TIMESTAMP_WITH_TIME_ZONE()),
      makeFlatVector<int32_t>({20, 40}));
  assertEqualVectors(expected, evaluate("map_except(c0, c1)", data));
}

TEST_F(MapExceptTest, flatMapConstantKeys) {
  auto input = makeFlatMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:null, 4:40}",
      "{1:11, 3:33}",
      "{}",
      "{2:22, 4:44}",
  });
  auto data = makeRowVector({input});

  auto result = evaluate("map_except(c0, array_constructor(1, 3, 5))", data);

  assertEqualVectors(
      makeFlatMapVectorFromJson<int64_t, int32_t>({
          "{2:20, 4:40}",
          "{}",
          "{}",
          "{2:22, 4:44}",
      }),
      result);

  // The excluded channels are dropped; the others keep the input's keys, map
  // values and in-map buffers.
  ASSERT_EQ(result->encoding(), VectorEncoding::Simple::FLAT_MAP);
  auto* flatResult = result->as<FlatMapVector>();
  EXPECT_EQ(flatResult->numDistinctKeys(), 2);
  EXPECT_EQ(
      flatResult->distinctKeys()->valueVector().get(),
      input->distinctKeys().get());
  for (const int64_t key : {2, 4}) {
    EXPECT_EQ(
        flatResult->mapValuesAt(*flatResult->getKeyChannel(key)).get(),
        input->mapValuesAt(*input->getKeyChannel(key)).get());
    EXPECT_EQ(
        flatResult->inMapsAt(*flatResult->getKeyChannel(key)).get(),
        input->inMapsAt(*input->getKeyChannel(key)).get());
  }

  // Nothing to exclude returns the input as is.
  EXPECT_EQ(
      evaluate("map_except(c0, array_constructor()::bigint[])", data).get(),
      input.get());
  EXPECT_EQ(
      evaluate("map_except(c0, array_constructor(7, 8))", data).get(),
      input.get());

  // Every key excluded.
  auto emptyResult =
      evaluate("map_except(c0, array_constructor(1, 2, 3, 4))", data);
  assertEqualVectors(
      makeFlatMapVectorFromJson<int64_t, int32_t>({"{}", "{}", "{}", "{}"}),
      emptyResult);
  ASSERT_EQ(emptyResult->encoding(), VectorEncoding::Simple::FLAT_MAP);
  EXPECT_EQ(emptyResult->as<FlatMapVector>()->numDistinctKeys(), 0);

  // Null and duplicate excluded keys.
  assertEqualVectors(
      makeFlatMapVectorFromJson<int64_t, int32_t>({
          "{2:20, 4:40}",
          "{}",
          "{}",
          "{2:22, 4:44}",
      }),
      evaluate(
          "map_except(c0, array_constructor(1, cast(null as bigint), 1, 3))",
          data));

  // An all-null constant list of keys makes every row null.
  auto nullResult = evaluate(
      "map_except(c0, c1)",
      makeRowVector(
          {input,
           BaseVector::createNullConstant(
               ARRAY(BIGINT()), input->size(), pool())}));
  for (vector_size_t row = 0; row < input->size(); ++row) {
    EXPECT_TRUE(nullResult->isNullAt(row)) << "row " << row;
  }

  // A constant list of keys may point at any row of the base array.
  assertEqualVectors(
      makeFlatMapVectorFromJson<int64_t, int32_t>({
          "{2:20, 4:40}",
          "{}",
          "{}",
          "{2:22, 4:44}",
      }),
      evaluate(
          "map_except(c0, c1)",
          makeRowVector({
              input,
              BaseVector::wrapInConstant(
                  input->size(),
                  1,
                  makeArrayVectorFromJson<int64_t>({"[7]", "[1, 3]"})),
          })));
}

TEST_F(MapExceptTest, flatMapNonConstantKeys) {
  auto input = makeFlatMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:null, 4:40}",
      "{1:11, 2:22, 3:33}",
      "{1:12, 2:23}",
      "{}",
  });
  auto data = makeRowVector({
      input,
      makeArrayVectorFromJson<int64_t>({
          "[1, 3]",
          "[2]",
          "[5]",
          "[1, 2]",
      }),
  });

  auto result = evaluate("map_except(c0, c1)", data);

  assertEqualVectors(
      makeFlatMapVectorFromJson<int64_t, int32_t>({
          "{2:20, 4:40}",
          "{1:11, 3:33}",
          "{1:12, 2:23}",
          "{}",
      }),
      result);

  // Keys, map values, the key index and the in-map buffers of channels no row
  // removes are shared with the input.
  ASSERT_EQ(result->encoding(), VectorEncoding::Simple::FLAT_MAP);
  auto* flatResult = result->as<FlatMapVector>();
  EXPECT_EQ(flatResult->distinctKeys().get(), input->distinctKeys().get());
  EXPECT_TRUE(flatResult->testingSharesKeyIndex(*input));
  for (column_index_t channel = 0; channel < input->numDistinctKeys();
       ++channel) {
    EXPECT_EQ(
        flatResult->mapValuesAt(channel).get(),
        input->mapValuesAt(channel).get());
  }
  const auto channel4 = *input->getKeyChannel(int64_t{4});
  EXPECT_EQ(
      flatResult->inMapsAt(channel4).get(), input->inMapsAt(channel4).get());
  const auto channel1 = *input->getKeyChannel(int64_t{1});
  EXPECT_NE(
      flatResult->inMapsAt(channel1).get(), input->inMapsAt(channel1).get());

  // Rows that remove nothing return the input as is.
  EXPECT_EQ(
      evaluate(
          "map_except(c0, c1)",
          makeRowVector({
              input,
              makeArrayVectorFromJson<int64_t>({"[5]", "[]", "[4]", "[1]"}),
          }))
          .get(),
      input.get());

  // Empty, null and duplicate excluded keys. A null list of keys makes the
  // whole row null.
  assertEqualVectors(
      makeFlatMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20, 3:null, 4:40}",
          "{3:33}",
          "{1:12}",
          "null",
      }),
      evaluate(
          "map_except(c0, c1)",
          makeRowVector({
              input,
              makeArrayVectorFromJson<int64_t>({
                  "[]",
                  "[1, null, 2, 1]",
                  "[2, 2]",
                  "null",
              }),
          })));
}

TEST_F(MapExceptTest, flatMapNulls) {
  auto data = makeRowVector({
      makeFlatMapVectorFromJson<int64_t, int32_t>({
          "{1:10, 2:20}",
          "null",
          "{1:11, 2:null}",
          "null",
      }),
      makeArrayVectorFromJson<int64_t>({
          "[1]",
          "[1]",
          "[1]",
          "[2]",
      }),
  });

  assertEqualVectors(
      makeFlatMapVectorFromJson<int64_t, int32_t>({
          "{2:20}",
          "null",
          "{2:null}",
          "null",
      }),
      evaluate("map_except(c0, c1)", data));

  assertEqualVectors(
      makeFlatMapVectorFromJson<int64_t, int32_t>({
          "{2:20}",
          "null",
          "{2:null}",
          "null",
      }),
      evaluate("map_except(c0, array_constructor(1))", data));
}

TEST_F(MapExceptTest, flatMapIntegerAndVarcharKeys) {
  auto integerData = makeRowVector({
      makeFlatMapVectorFromJson<int32_t, float>({
          "{1:1.5, 2:2.5, 3:3.5}",
          "{2:2.0, 3:3.0}",
      }),
      makeArrayVectorFromJson<int32_t>({"[2]", "[3, 9]"}),
  });
  assertEqualVectors(
      makeFlatMapVectorFromJson<int32_t, float>({
          "{1:1.5, 3:3.5}",
          "{2:2.0}",
      }),
      evaluate("map_except(c0, c1)", integerData));

  auto varcharData = makeRowVector({
      makeFlatMapVectorFromJson<std::string, int32_t>({
          "{\"apple\": 1, \"banana\": 2, \"Cucurbitaceae\": null}",
          "{\"banana\": 2, \"orange\": 4}",
      }),
      makeArrayVectorFromJson<std::string>({
          "[\"apple\", \"Cucurbitaceae\"]",
          "[\"orange\", \"apple\"]",
      }),
  });

  assertEqualVectors(
      makeFlatMapVectorFromJson<std::string, int32_t>({
          "{\"banana\": 2}",
          "{\"banana\": 2, \"orange\": 4}",
      }),
      evaluate(
          "map_except(c0, array_constructor('apple', 'Cucurbitaceae'))",
          varcharData));

  assertEqualVectors(
      makeFlatMapVectorFromJson<std::string, int32_t>({
          "{\"banana\": 2}",
          "{\"banana\": 2}",
      }),
      evaluate("map_except(c0, c1)", varcharData));
}

TEST_F(MapExceptTest, flatMapFloatNaNs) {
  // NaN keys match each other, as they do for a MapVector.
  static const auto kNaN = std::numeric_limits<double>::quiet_NaN();
  static const auto kSNaN = std::numeric_limits<double>::signaling_NaN();

  auto flatMap = makeFlatMapVectorFromJson<double, int32_t>({
      "{1:10, NaN:20, 3:null, 4:40}",
      "{NaN:21, 3:31}",
  });
  auto keys = makeArrayVector<double>({{1, kSNaN}, {3}});

  assertEqualVectors(
      evaluate(
          "map_except(c0, c1)", makeRowVector({flatMap->toMapVector(), keys})),
      evaluate("map_except(c0, c1)", makeRowVector({flatMap, keys})));
  assertEqualVectors(
      evaluate(
          "map_except(c0, c1)",
          makeRowVector(
              {flatMap->toMapVector(),
               BaseVector::wrapInConstant(
                   2, 0, makeArrayVector<double>({{kNaN, 4}}))})),
      evaluate(
          "map_except(c0, c1)",
          makeRowVector(
              {flatMap,
               BaseVector::wrapInConstant(
                   2, 0, makeArrayVector<double>({{kNaN, 4}}))})));

  auto indices = makeIndices({1, 0, 1});
  assertEqualVectors(
      evaluate(
          "map_except(c0, c1)",
          makeRowVector(
              {wrapInDictionary(indices, 3, flatMap->toMapVector()),
               makeArrayVector<double>({{kNaN}, {3}, {3}})})),
      evaluate(
          "map_except(c0, c1)",
          makeRowVector(
              {wrapInDictionary(indices, 3, flatMap),
               makeArrayVector<double>({{kNaN}, {3}, {3}})})));
}

TEST_F(MapExceptTest, flatMapWrappedEncodings) {
  auto input = makeFlatMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30}",
      "{1:11, 3:33}",
      "{2:22}",
      "{}",
      "{1:14, 2:24, 3:34}",
      "{3:35}",
      "{1:16, 2:26}",
      "{2:27, 3:37}",
      "{1:18}",
      "{1:19, 2:29, 3:39}",
  });
  auto expected = makeFlatMapVectorFromJson<int64_t, int32_t>({
      "{2:20}",
      "{}",
      "{2:22}",
      "{}",
      "{2:24}",
      "{}",
      "{2:26}",
      "{2:27}",
      "{}",
      "{2:29}",
  });

  auto scattered = [](auto row) { return (row * 17 + 3) % 10; };
  auto wrapped = wrapInDictionary(
      makeIndices(input->size(), scattered), input->size(), input);

  assertEqualVectors(
      wrapInDictionary(
          makeIndices(expected->size(), scattered), expected->size(), expected),
      evaluate(
          "map_except(c0, array_constructor(1, 3))", makeRowVector({wrapped})));

  // Non-constant excluded keys keep the dictionary from being peeled off, so
  // the function has to translate the wrapping itself.
  assertEqualVectors(
      wrapInDictionary(
          makeIndices(expected->size(), scattered), expected->size(), expected),
      evaluate(
          "map_except(c0, c1)",
          makeRowVector({
              wrapped,
              makeArrayVector<int64_t>(
                  input->size(),
                  [](auto /*row*/) { return 2; },
                  [](auto /*row*/, auto index) { return index == 0 ? 1 : 3; }),
          })));

  // Rows sharing a flat map row but excluding different keys.
  assertEqualVectors(
      makeFlatMapVectorFromJson<int64_t, int32_t>({
          "{2:20, 3:30}",
          "{1:10, 3:30}",
          "{1:10, 2:20}",
      }),
      evaluate(
          "map_except(c0, c1)",
          makeRowVector({
              wrapInDictionary(makeIndices({0, 0, 0}), 3, input),
              makeArrayVectorFromJson<int64_t>({"[1]", "[2]", "[3]"}),
          })));

  // Constant-encoded flat map. The constant is wider than the flat map it
  // wraps.
  assertEqualVectors(
      BaseVector::wrapInConstant(5, 1, expected),
      evaluate(
          "map_except(c0, array_constructor(1, 3))",
          makeRowVector({BaseVector::wrapInConstant(5, 1, input)})));

  // A dictionary narrower than the flat map it wraps, with per-row keys so the
  // wrapping is not peeled off. The result is built in the output row space,
  // which is smaller than the flat map's.
  const auto half = input->size() / 2;
  assertEqualVectors(
      wrapInDictionary(makeIndices(half, scattered), half, expected),
      evaluate(
          "map_except(c0, c1)",
          makeRowVector({
              wrapInDictionary(makeIndices(half, scattered), half, input),
              makeArrayVector<int64_t>(
                  half,
                  [](auto /*row*/) { return 2; },
                  [](auto /*row*/, auto index) { return index == 0 ? 1 : 3; }),
          })));
}

TEST_F(MapExceptTest, flatMapConstantKeysWrapped) {
  // Without peeling, as for batches under expression.min_rows_for_peeling, the
  // function sees the wrapped flat map with constant keys. The result drops the
  // excluded channels of the flat map and keeps the input's wrapping.
  queryCtx_->testingOverrideConfigUnsafe(
      {{core::QueryConfig::kDebugDisableExpressionWithPeeling, "true"}});

  auto input = makeFlatMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20, 3:30}",
      "{1:11, 3:33}",
      "{2:22}",
      "{1:13, 2:23}",
  });
  auto expected = makeFlatMapVectorFromJson<int64_t, int32_t>({
      "{2:20}",
      "{}",
      "{2:22}",
      "{2:23}",
  });
  const auto excluded = BaseVector::wrapInConstant(
      6, 0, makeArrayVectorFromJson<int64_t>({"[1, 3, 5]"}));

  auto indices = makeIndices({3, 0, 2, 2, 1, 0});
  auto nulls = makeNulls({false, false, true, false, false, false});
  for (const auto& [wrapped, wrappedExpected] :
       std::vector<std::pair<VectorPtr, VectorPtr>>{
           {BaseVector::wrapInDictionary(nullptr, indices, 6, input),
            BaseVector::wrapInDictionary(nullptr, indices, 6, expected)},
           {BaseVector::wrapInDictionary(nulls, indices, 6, input),
            BaseVector::wrapInDictionary(nulls, indices, 6, expected)},
           {BaseVector::wrapInConstant(6, 3, input),
            BaseVector::wrapInConstant(6, 3, expected)},
       }) {
    SCOPED_TRACE(wrapped->toString());
    auto result =
        evaluate("map_except(c0, c1)", makeRowVector({wrapped, excluded}));
    assertEqualVectors(wrappedExpected, result);

    auto* flatResult = result->wrappedVector()->as<FlatMapVector>();
    ASSERT_NE(flatResult, nullptr);
    ASSERT_EQ(flatResult->numDistinctKeys(), 1);
    const auto channel2 = *input->getKeyChannel(int64_t{2});
    EXPECT_EQ(
        flatResult->mapValuesAt(0).get(), input->mapValuesAt(channel2).get());
    EXPECT_EQ(flatResult->inMapsAt(0).get(), input->inMapsAt(channel2).get());
  }
}

TEST_F(MapExceptTest, flatMapDictionaryAddingNulls) {
  // A dictionary that maps every row to itself and only adds nulls, as
  // map_concat_bedrock returns for a null Hive row. Untouched in-map buffers
  // are still shared.
  auto input = makeFlatMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20}",
      "{1:11, 2:21}",
      "{1:12}",
  });
  auto wrapped = BaseVector::wrapInDictionary(
      makeNulls({false, true, false}), makeIndices({0, 1, 2}), 3, input);

  for (const auto& keys : std::vector<VectorPtr>{
           makeArrayVectorFromJson<int64_t>({"[1]", "[1]", "[3]"}),
           BaseVector::wrapInConstant(
               3, 0, makeArrayVectorFromJson<int64_t>({"[1]"}))}) {
    auto result =
        evaluate("map_except(c0, c1)", makeRowVector({wrapped, keys}));
    // The engine may peel the dictionary off when the keys are constant.
    auto* flatResult = result->wrappedVector()->as<FlatMapVector>();
    ASSERT_NE(flatResult, nullptr);
    const auto channel2 = *flatResult->getKeyChannel(int64_t{2});
    EXPECT_EQ(
        flatResult->inMapsAt(channel2).get(),
        input->inMapsAt(*input->getKeyChannel(int64_t{2})).get());
  }

  assertEqualVectors(
      makeFlatMapVectorFromJson<int64_t, int32_t>({"{2:20}", "null", "{1:12}"}),
      evaluate(
          "map_except(c0, c1)",
          makeRowVector(
              {wrapped,
               makeArrayVectorFromJson<int64_t>({"[1]", "[1]", "[3]"})})));
}

TEST_F(MapExceptTest, flatMapWrappedNullRowIndex) {
  // A dictionary leaves its indices unconstrained wherever it is null, so a
  // null row can carry an index past the end of the flat map. The wrapped map
  // values carry no nulls of their own, which makes every one of their indices
  // reachable and therefore required to be in range.
  auto input = makeFlatMapVectorFromJson<int64_t, int32_t>({
      "{1:10, 2:20}",
      "{1:11, 2:21}",
  });

  auto wrapped = BaseVector::wrapInDictionary(
      makeNulls({false, true, false}), makeIndices({1, 12345, 0}), 3, input);

  auto result = evaluate(
      "map_except(c0, c1)",
      makeRowVector({
          wrapped,
          makeArrayVectorFromJson<int64_t>({"[1]", "[1]", "[1]"}),
      }));

  auto* flatMap = result->as<FlatMapVector>();
  ASSERT_NE(flatMap, nullptr);
  for (column_index_t channel = 0; channel < flatMap->numDistinctKeys();
       ++channel) {
    flatMap->mapValuesAt(channel)->validate();
  }
}

TEST_F(MapExceptTest, flatMapTimestampWithTimeZoneKey) {
  // The flat map key lookup compares keys by instant, regardless of time zone.
  auto flatMap = std::make_shared<FlatMapVector>(
      pool(),
      MAP(TIMESTAMP_WITH_TIME_ZONE(), INTEGER()),
      nullptr,
      2,
      makeFlatVector<int64_t>(
          {pack(1, 1), pack(2, 2), pack(3, 3)}, TIMESTAMP_WITH_TIME_ZONE()),
      std::vector<VectorPtr>{
          makeFlatVector<int32_t>({10, 11}),
          makeFlatVector<int32_t>({20, 21}),
          makeFlatVector<int32_t>({30, 31})},
      std::vector<BufferPtr>{});
  auto map = flatMap->toMapVector();

  auto excluded = makeFlatVector<int64_t>(
      {pack(1, 7), pack(3, 8), pack(2, 9)}, TIMESTAMP_WITH_TIME_ZONE());
  for (const auto& keys : std::vector<VectorPtr>{
           BaseVector::wrapInConstant(2, 0, makeArrayVector({0}, excluded)),
           makeArrayVector({0, 2}, excluded)}) {
    auto result =
        evaluate("map_except(c0, c1)", makeRowVector({flatMap, keys}));
    ASSERT_EQ(result->encoding(), VectorEncoding::Simple::FLAT_MAP);
    assertEqualVectors(
        evaluate("map_except(c0, c1)", makeRowVector({map, keys})), result);
  }
}

TEST_F(MapExceptTest, fuzzFlatMap) {
  VectorFuzzer::Options options;
  options.allowFlatMapVector = true;
  VectorFuzzer fuzzer(options, pool());

  constexpr vector_size_t kSize = 100;

  for (auto iteration = 0; iteration < 50; ++iteration) {
    SCOPED_TRACE(fmt::format("iteration: {}", iteration));

    auto flatMap = fuzzer.fuzzFlatMap(BIGINT(), INTEGER(), kSize);
    auto map = flatMap->toMapVector();

    // Exclude about half of the keys the flat map actually holds, so the
    // result is neither everything nor nothing.
    const auto* distinctKeys =
        flatMap->distinctKeys()->as<SimpleVector<int64_t>>();
    std::vector<int64_t> excludedKeys;
    for (vector_size_t i = 0; i < flatMap->numDistinctKeys(); i += 2) {
      excludedKeys.push_back(distinctKeys->valueAt(i));
    }

    const auto keySizeAt = [&](vector_size_t row) {
      return excludedKeys.empty() ? 0 : row % (excludedKeys.size() + 1);
    };
    const auto keyAt = [&](vector_size_t row, vector_size_t index) {
      return excludedKeys[(row + index) % excludedKeys.size()];
    };

    std::vector<std::pair<std::string, VectorPtr>> keyArguments = {
        {"constant keys",
         BaseVector::wrapInConstant(
             kSize, 0, makeArrayVector<int64_t>({excludedKeys}))},
        {"per-row keys",
         makeArrayVector<int64_t>(kSize, keySizeAt, keyAt, nullEvery(7))},
    };

    const auto scattered = [](auto row) { return (row * 37 + 11) % kSize; };
    std::vector<std::pair<std::string, VectorPtr>> mapArguments = {
        {"flat map", flatMap},
        {"wrapped flat map",
         wrapInDictionary(makeIndices(kSize, scattered), kSize, flatMap)},
    };

    for (const auto& [mapName, mapArgument] : mapArguments) {
      SCOPED_TRACE(mapName);
      const auto expectedMap = mapArgument == flatMap
          ? VectorPtr(map)
          : wrapInDictionary(makeIndices(kSize, scattered), kSize, map);
      for (const auto& [name, keys] : keyArguments) {
        SCOPED_TRACE(name);
        assertEqualVectors(
            evaluate("map_except(c0, c1)", makeRowVector({expectedMap, keys})),
            evaluate("map_except(c0, c1)", makeRowVector({mapArgument, keys})));
      }
    }
  }
}

// Custom fuzzer tests

// Custom fuzzer tests to compare map_except with equivalent expression
// using existing UDFs. The equivalent expression is:
// map_filter(map, (k, v) -> NOT contains(keys_array, k))
//
// Note: We already have explicit tests for NULL handling in MapExceptTest.
// These fuzzer tests focus on comparing behavior with non-null inputs.
class MapExceptFuzzerTest : public test::FunctionBaseTest {
 protected:
  // Helper to flatten vectors for consistent comparison across encodings
  template <typename T>
  static VectorPtr flatten(const std::shared_ptr<T>& vector) {
    SelectivityVector allRows(vector->size());
    auto result =
        BaseVector::create(vector->type(), vector->size(), vector->pool());
    result->copy(vector.get(), allRows, nullptr);
    return result;
  }

  // The equivalent SQL expression for map_except using existing UDFs.
  // map_except(map, keys_array) is equivalent to:
  // map_filter(map, (k, v) -> NOT coalesce(contains(keys_array, k), false))
  //
  // Note: We use coalesce to handle the case where contains returns null
  // (when comparing with null elements). map_except ignores null keys in
  // the exclusion array, so we treat null as "not found" (false).
  static constexpr const char* kEquivalentExpression =
      "map_filter(c0, (k, v) -> NOT coalesce(contains(c1, k), false))";

  // Get a SelectivityVector that excludes rows where either input is null.
  // This is because map_except returns NULL when either input is NULL.
  static SelectivityVector getNonNullRows(const RowVectorPtr& data) {
    auto inputMap = data->childAt(0);
    auto keysArray = data->childAt(1);
    SelectivityVector nonNullRows(data->size());

    for (vector_size_t i = 0; i < data->size(); ++i) {
      if (inputMap->isNullAt(i) || keysArray->isNullAt(i)) {
        nonNullRows.setValid(i, false);
      }
    }
    nonNullRows.updateBounds();
    return nonNullRows;
  }

  void testEquivalence(const RowVectorPtr& data) {
    auto result = evaluate("map_except(c0, c1)", data);
    auto expected = evaluate(kEquivalentExpression, data);

    // Get rows where neither input is null
    auto nonNullRows = getNonNullRows(data);

    // Compare only non-null rows (null propagation is tested separately)
    for (auto i = 0; i < data->size(); ++i) {
      if (nonNullRows.isValid(i)) {
        ASSERT_TRUE(expected->equalValueAt(result.get(), i, i))
            << "Mismatch at row " << i << ": expected " << expected->toString(i)
            << ", got " << result->toString(i);
      }
    }
  }

  void runFuzzTest(
      const TypePtr& keyType,
      const TypePtr& valueType,
      const test::FuzzerTestOptions& opts) {
    test::FuzzerTestHelper helper(pool());
    helper.runMapArrayTest(
        keyType,
        valueType,
        [this](const VectorPtr& inputMap, const VectorPtr& keysArray) {
          auto data = makeRowVector({inputMap, keysArray});
          testEquivalence(data);
        },
        opts);
  }
};

// Fuzz test with flat vectors, no nulls, fixed-size maps
TEST_F(MapExceptFuzzerTest, fuzzFlatNoNulls) {
  runFuzzTest(
      INTEGER(),
      INTEGER(),
      {.vectorSize = 100,
       .nullRatio = 0.0,
       .containerLength = 10,
       .iterations = 10});
}

// Fuzz test with nulls in maps and keys array
TEST_F(MapExceptFuzzerTest, fuzzWithNulls) {
  runFuzzTest(
      INTEGER(),
      INTEGER(),
      {.vectorSize = 100,
       .containerLength = 10,
       .containerHasNulls = true,
       .containerVariableLength = true,
       .iterations = 10});
}

// Fuzz test with dictionary-encoded vectors
TEST_F(MapExceptFuzzerTest, fuzzDictionaryEncoded) {
  test::FuzzerTestHelper helper(pool());
  test::FuzzerTestOptions opts{
      .vectorSize = 100,
      .containerLength = 10,
      .containerHasNulls = true,
      .containerVariableLength = true};
  auto fuzzer = helper.createFuzzer(opts);

  for (auto i = 0; i < 10; ++i) {
    auto baseInputMap = fuzzer.fuzz(MAP(INTEGER(), INTEGER()));
    auto baseKeysArray = fuzzer.fuzz(ARRAY(INTEGER()));

    auto inputMap = fuzzer.fuzzDictionary(baseInputMap, opts.vectorSize);
    auto keysArray = fuzzer.fuzzDictionary(baseKeysArray, opts.vectorSize);

    auto data = makeRowVector({inputMap, keysArray});
    auto flatData = makeRowVector({flatten(inputMap), flatten(keysArray)});

    auto result = evaluate("map_except(c0, c1)", data);
    auto expectedResult = evaluate("map_except(c0, c1)", flatData);
    assertEqualVectors(expectedResult, result);

    testEquivalence(flatData);
  }
}

// Fuzz test with variable-length maps and high null ratio
TEST_F(MapExceptFuzzerTest, fuzzVariableLengthWithHighNullRatio) {
  runFuzzTest(
      INTEGER(),
      INTEGER(),
      {.vectorSize = 100,
       .nullRatio = 0.3,
       .containerLength = 20,
       .containerHasNulls = true,
       .containerVariableLength = true,
       .iterations = 10});
}

// Fuzz test with string keys
TEST_F(MapExceptFuzzerTest, fuzzStringKeys) {
  runFuzzTest(
      VARCHAR(),
      INTEGER(),
      {.vectorSize = 100,
       .containerLength = 10,
       .containerHasNulls = true,
       .containerVariableLength = true,
       .iterations = 10});
}

// Fuzz test with bigint keys
TEST_F(MapExceptFuzzerTest, fuzzBigintKeys) {
  runFuzzTest(
      BIGINT(),
      VARCHAR(),
      {.vectorSize = 100,
       .containerLength = 10,
       .containerHasNulls = true,
       .containerVariableLength = true,
       .iterations = 10});
}

// Fuzz test with empty maps and arrays
TEST_F(MapExceptFuzzerTest, fuzzWithEmptyContainers) {
  runFuzzTest(
      INTEGER(),
      INTEGER(),
      {.vectorSize = 100,
       .containerLength = 2,
       .containerHasNulls = true,
       .containerVariableLength = true,
       .iterations = 10});
}

// Fuzz test with various value types
TEST_F(MapExceptFuzzerTest, fuzzVariousValueTypes) {
  test::FuzzerTestOptions opts{
      .vectorSize = 100,
      .containerLength = 10,
      .containerHasNulls = true,
      .containerVariableLength = true,
      .iterations = 5};

  // Test with DOUBLE values
  runFuzzTest(INTEGER(), DOUBLE(), opts);

  // Test with BOOLEAN values
  runFuzzTest(INTEGER(), BOOLEAN(), opts);

  // Test with VARCHAR values
  runFuzzTest(INTEGER(), VARCHAR(), opts);
}

// Fuzz test with constant vectors
TEST_F(MapExceptFuzzerTest, fuzzConstantVectors) {
  test::FuzzerTestHelper helper(pool());
  test::FuzzerTestOptions opts{
      .vectorSize = 100, .nullRatio = 0.0, .containerLength = 5};
  auto fuzzer = helper.createFuzzer(opts);

  for (auto i = 0; i < 10; ++i) {
    auto inputMap = fuzzer.fuzz(MAP(INTEGER(), INTEGER()));
    auto keysArray = fuzzer.fuzzConstant(ARRAY(INTEGER()), opts.vectorSize);

    auto data = makeRowVector({inputMap, keysArray});
    testEquivalence(data);
  }
}

// Fuzz test with array keys, which take the generic path.
TEST_F(MapExceptFuzzerTest, fuzzArrayKeys) {
  runFuzzTest(
      ARRAY(INTEGER()),
      INTEGER(),
      {.vectorSize = 100,
       .nullRatio = 0.0,
       .containerLength = 3,
       .containerVariableLength = true,
       .iterations = 10});
}

// Stress test with large vectors
TEST_F(MapExceptFuzzerTest, fuzzLargeVectors) {
  runFuzzTest(
      INTEGER(),
      INTEGER(),
      {.vectorSize = 1000,
       .containerLength = 50,
       .containerHasNulls = true,
       .containerVariableLength = true,
       .iterations = 5});
}

} // namespace
} // namespace facebook::velox::functions
