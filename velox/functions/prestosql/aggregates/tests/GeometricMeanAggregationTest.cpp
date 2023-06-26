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
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/AggregationTestBase.h"
#include "velox/functions/prestosql/aggregates/DecimalAggregate.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::test {

namespace {

class GeometricMeanAggregationTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    allowInputShuffle();
  }

  RowVectorPtr rowVector_ = makeRowVector(
      {makeNullableFlatVector<int64_t>(
           {1737832077492805767,
            4077358421272316858,
            8352305779871864636,
            std::nullopt,
            65277021944352935,
            5057378477121420109,
            790130315336096174,
            3724289589169410468,
            2074604147183559854,
            3833098290310622035}),
       makeNullableFlatVector<int16_t>(
           {24756,
            std::nullopt,
            21428,
            27045,
            16834,
            26164,
            11710,
            30375,
            29488,
            std::nullopt}),
       makeFlatVector<int32_t>(
           {198304612,
            417177801,
            758242871,
            537655879,
            1014685970,
            2107063880,
            1444803288,
            540721923,
            219972873,
            1755486969}),
       makeNullableFlatVector<int64_t>(
           {6524373357247204968,
            1459602477200235160,
            std::nullopt,
            5427507077629018454,
            6362318851342815124,
            6567761115475435067,
            9193194088128540374,
            7862838580565801772,
            7650459730033994045,
            327870505158904254}),
       makeFlatVector<float>(
           {0.25509512424468994,
            0.9400740265846252,
            0.8909032344818115,
            0.5809566974639893,
            0.257508248090744,
            0.9885215759277344,
            0.8142848014831543,
            0.013539127074182034,
            0.34998375177383423,
            0.8484677672386169}),
       makeNullableFlatVector<double>(
           {0.7788977100218852,
            0.35165951193021494,
            0.594503561162791,
            0.6952328836279545,
            0.7572002300043404,
            0.5615574422712544,
            0.5273714584940704,
            std::nullopt,
            0.5928238785777165,
            0.964966371918541})});
};

// All expected results taken from Presto
TEST_F(GeometricMeanAggregationTest, geoMeanConst) {
  auto vectors = {
      makeRowVector({
          makeFlatVector<int64_t>(
              10, [](vector_size_t row) { return row / 3; }),
          makeConstant(5, 10),
          makeConstant(6.0, 10),
      }),
      makeRowVector({
          makeFlatVector<int64_t>(
              10, [](vector_size_t row) { return row / 3; }),
          makeConstant(5, 10),
          makeConstant(6.0, 10),
      }),
  };

  testAggregations(
      vectors,
      {},
      {"geometric_mean(c1)", "geometric_mean(c2)"},
      "SELECT 4.999999999999998, 6.0");

  testAggregations(
      {makeRowVector({makeFlatVector<int64_t>({7, 9, 12})})},
      {},
      {"geometric_mean(c0)"},
      {makeRowVector(
          {makeConstant(std::optional<double>(9.109766915626988), 1)})});

  auto resultVectorGrpBy1 = {makeRowVector(
      {makeFlatVector<int64_t>({0, 1, 2, 3}),
       makeFlatVector<double>(
           {4.999999999999999,
            4.999999999999999,
            4.999999999999999,
            4.999999999999999}),
       makeFlatVector<double>(
           {5.999999999999998, 5.999999999999998, 5.999999999999998, 6})})};

  testAggregations(
      vectors,
      {"c0"},
      {"geometric_mean(c1)", "geometric_mean(c2)"},
      resultVectorGrpBy1);

  testAggregations(vectors, {}, {"geometric_mean(c0)"}, "SELECT 0.0");

  auto resultVectorGrpBy2 = {makeRowVector({
      makeFlatVector<int64_t>({0, 1}),
      makeFlatVector<double>({0.0, 1.3160740129524926}),
  })};

  testAggregations(
      [&](auto& builder) {
        builder.values(vectors).project({"c0 % 2 AS c0_mod_2", "c0"});
      },
      {"c0_mod_2"},
      {"geometric_mean(c0)"},
      {},
      [&](auto& builder) { return builder.assertResults(resultVectorGrpBy2); });
}

TEST_F(GeometricMeanAggregationTest, geoMeanConstNull) {
  // Have at least two row vectors as it triggers different code paths.
  auto vectors = {
      makeRowVector({
          makeNullableFlatVector<int64_t>({0, 1, 2, 0, 1, 2, 0, 1, 2, 0}),
          makeNullConstant(TypeKind::BIGINT, 10),
          makeNullConstant(TypeKind::DOUBLE, 10),
      }),
      makeRowVector({
          makeNullableFlatVector<int64_t>({0, 1, 2, 0, 1, 2, 0, 1, 2, 0}),
          makeNullConstant(TypeKind::BIGINT, 10),
          makeNullConstant(TypeKind::DOUBLE, 10),
      }),
  };

  testAggregations(
      vectors,
      {},
      {"geometric_mean(c1)", "geometric_mean(c2)"},
      "SELECT null, null");

  auto resultVectorGrpBy1 = {makeRowVector(
      {makeFlatVector<int64_t>({0, 1, 2}),
       makeNullConstant(TypeKind::DOUBLE, 3),
       makeNullConstant(TypeKind::DOUBLE, 3)})};

  testAggregations(
      vectors,
      {"c0"},
      {"geometric_mean(c1)", "geometric_mean(c2)"},
      resultVectorGrpBy1);

  testAggregations(vectors, {}, {"geometric_mean(c0)"}, "SELECT 0.0");
}

TEST_F(GeometricMeanAggregationTest, geoMeanNulls) {
  // Have two row vectors a lest as it triggers different code paths.
  auto vectors = {
      makeRowVector({
          makeNullableFlatVector<int64_t>({0, std::nullopt, 2, 0, 1}),
          makeNullableFlatVector<int64_t>({0, 1, std::nullopt, 3, 4}),
          makeNullableFlatVector<double>({0.1, 1.2, 2.3, std::nullopt, 4.4}),
      }),
      makeRowVector({
          makeNullableFlatVector<int64_t>({0, std::nullopt, 2, 0, 1}),
          makeNullableFlatVector<int64_t>({0, 1, std::nullopt, 3, 4}),
          makeNullableFlatVector<double>({0.1, 1.2, 2.3, std::nullopt, 4.4}),
      }),
  };

  testAggregations(
      vectors,
      {},
      {"geometric_mean(c1)", "geometric_mean(c2)"},
      "SELECT 0.0, 1.0497610133342126");

  auto resultVectorGrpBy1 = {makeRowVector({
      makeNullableFlatVector<int64_t>({0, 1, 2, std::nullopt}),
      makeNullableFlatVector<double>({0.0, 4.0, std::nullopt, 1.0}),
      makeNullableFlatVector<double>({0.1, 4.4, 2.3, 1.2}),
  })};

  testAggregations(
      vectors,
      {"c0"},
      {"geometric_mean(c1)", "geometric_mean(c2)"},
      resultVectorGrpBy1);
}

TEST_F(GeometricMeanAggregationTest, geoMean) {
  auto vectors = {
      rowVector_,
      rowVector_,
      rowVector_,
      rowVector_,
      rowVector_,
      rowVector_,
      rowVector_,
      rowVector_,
      rowVector_,
      rowVector_};

  // global aggregation
  testAggregations(
      vectors,
      {},
      {"geometric_mean(c1)",
       "geometric_mean(c2)",
       "geometric_mean(c4)",
       "geometric_mean(c5)"},
      "SELECT 22525.26362949741, 6.819500309342173E8, 0.39945385, 0.6248603925324049");

  // global aggregation; no input
  testAggregations(
      [&](auto& builder) { builder.values(vectors).filter("c0 % 2 = 5"); },
      {},
      {"geometric_mean(c0)"},
      "SELECT null");

  // global aggregation over filter
  testAggregations(
      [&](auto& builder) { builder.values(vectors).filter("c0 % 5 = 3"); },
      {},
      {"geometric_mean(c1)"},
      "SELECT 30374.999999999953");

  auto resultVectorGrpBy = {makeRowVector(
      {makeNullableFlatVector<int64_t>({6, std::nullopt, 5, 4, 9, 8, 7}),
       makeFlatVector<double>(
           {21428.000000000004,
            27044.999999999996,
            16834.0,
            18582.37013946282,
            26164.00000000001,
            30374.999999999953,
            24755.999999999985}),
       makeFlatVector<double>(
           {7.582428709999994E8,
            5.376558789999989E8,
            1.3346415241412628E9,
            5.637530755403515E8,
            2.107063879999991E9,
            4.749496634272213E8,
            1.9830461200000027E8}),
       makeNullableFlatVector<double>(
           {std::nullopt,
            5.4275070776289874E18,
            1.4443049178659584E18,
            8.3864271988502139E18,
            6.5677611154754929E18,
            3.3877158484765978E18,
            6.5243733572471603E18}),
       makeFlatVector<float>(
           {0.89090323,
            0.58095676,
            0.46742642,
            0.5338412,
            0.9885216,
            0.11281748,
            0.25509512}),
       makeFlatVector<double>(
           {0.594503561162791,
            0.6952328836279544,
            0.8547939861529052,
            0.5591407635610571,
            0.5615574422712544,
            0.351659511930215,
            0.7788977100218851})})};

  // group by
  testAggregations(
      [&](auto& builder) {
        builder.values(vectors).project(
            {"c0 % 10 AS c0_mod_10", "c1", "c2", "c3", "c4", "c5"});
      },
      {"c0_mod_10"},
      {"geometric_mean(c1)",
       "geometric_mean(c2)",
       "geometric_mean(c3)",
       "geometric_mean(c4)",
       "geometric_mean(c5)"},
      {},
      [&](auto& builder) { return builder.assertResults(resultVectorGrpBy); });

  // group by; no input
  testAggregations(
      [&](auto& builder) {
        builder.values(vectors)
            .project({"c0 % 10 AS c0_mod_10", "c1"})
            .filter("c0_mod_10 > 10");
      },
      {"c0_mod_10"},
      {"geometric_mean(c1)"},
      "");

  // group by over filter
  testAggregations(
      [&](auto& builder) {
        builder.values(vectors)
            .filter("c2 % 5 = 3")
            .project({"c0 % 10 AS c0_mod_10", "c1"});
      },
      {"c0_mod_10"},
      {"geometric_mean(c1)"},
      "SELECT * from (VALUES (4, 18582.37013946282), (8, 30374.999999999953))");
}

TEST_F(GeometricMeanAggregationTest, partialResults) {
  auto data = makeRowVector(
      {makeFlatVector<int64_t>(10, [](auto row) { return row + 1; })});
  auto plan = PlanBuilder()
                  .values({data})
                  .partialAggregation({}, {"geometric_mean(c0)"})
                  .planNode();

  assertQuery(plan, "SELECT row(6.559763032876794, 10)");
}

TEST_F(GeometricMeanAggregationTest, constantVectorOverflow) {
  auto rows = makeRowVector({makeConstant<int32_t>(1073741824, 100)});
  auto plan = PlanBuilder()
                  .values({rows})
                  .singleAggregation({}, {"geometric_mean(c0)"})
                  .planNode();
  assertQuery(plan, "SELECT 1073741824");
}

} // namespace
} // namespace facebook::velox::aggregate::test
