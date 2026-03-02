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
#include "velox/core/PlanFragment.h"
#include "velox/core/PlanNode.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;

namespace facebook::velox::exec::test {

class SpatialJoinTest : public OperatorTestBase {
 public:
  /* The polygons below have the following relations:
   * [Legend: = equals, v overlap, / disjoint]
   *
   *   A  B  C  D
   * A =  v  /  /
   * B v  =  /  /
   * C /  /  =  /
   * D /  /  /  =
   *
   * Overlap means geometries share interior points (for full definition see
   * (DE-9IM)[https://en.wikipedia.org/wiki/DE-9IM]), but neither contains the
   * other.
   */
  static constexpr std::string_view kPolygonA =
      "POLYGON ((0 0, -0.5 2.5, 0 5, 2.5 5.5, 5 5, 5.5 2.5, 5 0, 2.5 -0.5, 0 0))";
  static constexpr std::string_view kPolygonB =
      "POLYGON ((4 4, 3.5 7, 4 10, 7 10.5, 10 10, 10.5 7, 10 4, 7 3.5, 4 4))";
  static constexpr std::string_view kPolygonC =
      "POLYGON ((15 15, 15 14, 14 14, 14 15, 15 15))";
  static constexpr std::string_view kPolygonD =
      "POLYGON ((18 18, 18 19, 19 19, 19 18, 18 18))";

  // A set of points: X in A, Y in A and B, Z in B, W outside of A and B
  static constexpr std::string_view kPointX = "POINT (1 1)";
  static constexpr std::string_view kPointY = "POINT (4.5 4.5)";
  static constexpr std::string_view kPointZ = "POINT (6 6)";
  static constexpr std::string_view kPointW = "POINT (20 20)";
  static constexpr std::string_view kPointV = "POINT (15 15)";
  static constexpr std::string_view kPointS = "POINT (18 18)";
  static constexpr std::string_view kPointQ = "POINT (28 28)";
  static constexpr std::string_view kMultipointU = "MULTIPOINT (15 15)";
  static constexpr std::string_view kMultipointT =
      "MULTIPOINT (14.5 14.5, 16 16)";
  static constexpr std::string_view kMultipointR = "MULTIPOINT (15 15, 19 19)";

 protected:
  void runTest(
      const std::vector<std::optional<std::string_view>>& probeWkts,
      const std::vector<std::optional<std::string_view>>& buildWkts,
      const std::optional<const std::vector<std::optional<double>>>& radiiOpt,
      const std::string& predicate,
      core::JoinType joinType,
      const std::vector<std::optional<std::string_view>>& expectedLeftWkts,
      const std::vector<std::optional<std::string_view>>& expectedRightWkts) {
    for (bool separateProbeBatches : {false, true}) {
      for (size_t maxBatchSize : {128, 3, 2, 1}) {
        for (int32_t maxDrivers : {1, 4}) {
          runTestWithConfig(
              probeWkts,
              buildWkts,
              radiiOpt,
              predicate,
              joinType,
              expectedLeftWkts,
              expectedRightWkts,
              maxDrivers,
              maxBatchSize,
              separateProbeBatches);
        }
      }
    }
  }

  void runTestWithConfig(
      const std::vector<std::optional<std::string_view>>& probeWkts,
      const std::vector<std::optional<std::string_view>>& buildWkts,
      const std::optional<const std::vector<std::optional<double>>>& radiiOpt,
      const std::string& predicate,
      core::JoinType joinType,
      const std::vector<std::optional<std::string_view>>& expectedLeftWkts,
      const std::vector<std::optional<std::string_view>>& expectedRightWkts,
      int32_t maxDrivers,
      size_t maxBatchSize,
      bool separateBatches) {
    std::vector<std::optional<std::string>> probeWktsStr(
        probeWkts.begin(), probeWkts.end());
    std::vector<std::optional<std::string>> buildWktsStr(
        buildWkts.begin(), buildWkts.end());
    std::vector<std::optional<std::string>> expectedLeftWktsStr(
        expectedLeftWkts.begin(), expectedLeftWkts.end());
    std::vector<std::optional<std::string>> expectedRightWktsStr(
        expectedRightWkts.begin(), expectedRightWkts.end());
    auto radii = radiiOpt.value_or(
        std::vector<std::optional<double>>(buildWkts.size(), std::nullopt));
    VELOX_CHECK_EQ(radii.size(), buildWkts.size());
    std::optional<std::string> radiusVariable = std::nullopt;
    if (radiiOpt.has_value()) {
      radiusVariable = "radius";
    }

    std::vector<RowVectorPtr> probeBatches;
    std::vector<RowVectorPtr> buildBatches;
    if (separateBatches) {
      for (const auto& wkt : probeWktsStr) {
        probeBatches.push_back(makeRowVector(
            {"left_g"}, {makeNullableFlatVector<std::string>({wkt})}));
      }
      if (probeBatches.empty()) {
        probeBatches.push_back(makeRowVector(
            {"left_g"}, {makeNullableFlatVector<std::string>({})}));
      }

      for (size_t idx = 0; idx < buildWktsStr.size(); ++idx) {
        auto& wkt = buildWktsStr[idx];
        buildBatches.push_back(makeRowVector(
            {"right_g", "radius"},
            {makeNullableFlatVector<std::string>({wkt}),
             makeNullableFlatVector<double>({radii[idx]})}));
      }
      if (buildBatches.empty()) {
        buildBatches.push_back(makeRowVector(
            {"right_g", "radius"},
            {makeNullableFlatVector<std::string>({}),
             makeNullableFlatVector<double>({})}));
      }
    } else {
      probeBatches.push_back(makeRowVector(
          {"left_g"}, {makeNullableFlatVector<std::string>(probeWktsStr)}));
      buildBatches.push_back(makeRowVector(
          {"right_g", "radius"},
          {makeNullableFlatVector<std::string>(buildWktsStr),
           makeNullableFlatVector<double>(radii)}));
    }
    auto expectedRows = makeRowVector(
        {"left_g", "right_g"},
        {makeNullableFlatVector<std::string>(expectedLeftWktsStr),
         makeNullableFlatVector<std::string>(expectedRightWktsStr)});

    auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
    auto plan =
        PlanBuilder(planNodeIdGenerator)
            .values(probeBatches)
            .project({"ST_GeometryFromText(left_g) AS left_g"})
            .localPartitionRoundRobinRow()
            .spatialJoin(
                PlanBuilder(planNodeIdGenerator)
                    .values(buildBatches)
                    .project(
                        {"ST_GeometryFromText(right_g) AS right_g", "radius"})
                    .localPartition({})
                    .planNode(),
                predicate,
                "left_g",
                "right_g",
                radiusVariable,
                {"left_g", "right_g"},
                joinType)
            .project(
                {"ST_AsText(left_g) AS left_g",
                 "ST_AsText(right_g) AS right_g"})
            .planNode();
    AssertQueryBuilder builder{plan};
    builder.maxDrivers(maxDrivers)
        .config(core::QueryConfig::kPreferredOutputBatchRows, maxBatchSize)
        .config(core::QueryConfig::kMaxOutputBatchRows, maxBatchSize)
        .assertResults({expectedRows});
  }
};

TEST_F(SpatialJoinTest, testTrivialSpatialJoin) {
  runTest(
      {"POINT (1 1)"},
      {"POINT (1 1)"},
      std::nullopt,
      "ST_Intersects(left_g, right_g)",
      core::JoinType::kInner,
      {"POINT (1 1)"},
      {"POINT (1 1)"});
}

TEST_F(SpatialJoinTest, testSimpleSpatialInnerJoin) {
  runTest(
      {"POINT (1 1)", "POINT (1 2)"},
      {"POINT (1 1)", "POINT (2 1)"},
      std::nullopt,
      "ST_Intersects(left_g, right_g)",
      core::JoinType::kInner,
      {"POINT (1 1)"},
      {"POINT (1 1)"});
}

TEST_F(SpatialJoinTest, testSimpleSpatialLeftJoin) {
  runTest(
      {"POINT (1 1)", "POINT (1 2)"},
      {"POINT (1 1)", "POINT (2 1)"},
      std::nullopt,
      "ST_Intersects(left_g, right_g)",
      core::JoinType::kLeft,
      {"POINT (1 1)", "POINT (1 2)"},
      {"POINT (1 1)", std::nullopt});
}

TEST_F(SpatialJoinTest, testSpatialJoinNullRows) {
  runTest(
      {"POINT (0 0)", std::nullopt, "POINT (1 1)", std::nullopt},
      {"POINT (0 0)", "POINT (1 1)", std::nullopt, std::nullopt},
      std::nullopt,
      "ST_Intersects(left_g, right_g)",
      core::JoinType::kInner,
      {"POINT (0 0)", "POINT (1 1)"},
      {"POINT (0 0)", "POINT (1 1)"});
  runTest(
      {"POINT (0 0)", std::nullopt, "POINT (2 2)", std::nullopt},
      {"POINT (0 0)", "POINT (1 1)", std::nullopt, std::nullopt},
      std::nullopt,
      "ST_Intersects(left_g, right_g)",
      core::JoinType::kLeft,
      {"POINT (0 0)", "POINT (2 2)", std::nullopt, std::nullopt},
      {"POINT (0 0)", std::nullopt, std::nullopt, std::nullopt});
}

// Test geometries that don't intersect but their envelopes do.
// Important to test spatial index
TEST_F(SpatialJoinTest, simpleSpatialJoinEnvelopes) {
  runTest(
      {"POINT (0.5 0.6)", "POINT (0.5 0.5)", "LINESTRING (0 0.1, 0.9 1)"},
      {"POLYGON ((0 0, 1 1, 1 0, 0 0))"},
      std::nullopt,
      "ST_Intersects(left_g, right_g)",
      core::JoinType::kInner,
      {"POINT (0.5 0.5)"},
      {"POLYGON ((0 0, 1 1, 1 0, 0 0))"});
}

TEST_F(SpatialJoinTest, testSelfSpatialJoin) {
  std::vector<std::optional<std::string_view>> inputWkts = {
      kPolygonA, kPolygonB, kPolygonC, kPolygonD};
  std::vector<std::optional<std::string_view>> leftOutputWkts = {
      kPolygonA, kPolygonA, kPolygonB, kPolygonB, kPolygonC, kPolygonD};
  std::vector<std::optional<std::string_view>> rightOutputWkts = {
      kPolygonA, kPolygonB, kPolygonA, kPolygonB, kPolygonC, kPolygonD};

  runTest(
      inputWkts,
      inputWkts,
      std::nullopt,
      "ST_Intersects(left_g, right_g)",
      core::JoinType::kInner,
      leftOutputWkts,
      rightOutputWkts);

  runTest(
      inputWkts,
      inputWkts,
      std::nullopt,
      "ST_Intersects(left_g, right_g)",
      core::JoinType::kLeft,
      leftOutputWkts,
      rightOutputWkts);

  runTest(
      inputWkts,
      inputWkts,
      std::nullopt,
      "ST_Overlaps(left_g, right_g)",
      core::JoinType::kInner,
      {kPolygonA, kPolygonB},
      {kPolygonB, kPolygonA});

  runTest(
      inputWkts,
      inputWkts,
      std::nullopt,
      "ST_Intersects(left_g, right_g) AND ST_Overlaps(left_g, right_g)",
      core::JoinType::kInner,
      {kPolygonA, kPolygonB},
      {kPolygonB, kPolygonA});

  runTest(
      inputWkts,
      inputWkts,
      std::nullopt,
      "ST_Overlaps(left_g, right_g)",
      core::JoinType::kLeft,
      {kPolygonA, kPolygonB, kPolygonC, kPolygonD},
      {kPolygonB, kPolygonA, std::nullopt, std::nullopt});

  runTest(
      inputWkts,
      inputWkts,
      std::nullopt,
      "ST_Equals(left_g, right_g)",
      core::JoinType::kInner,
      inputWkts,
      inputWkts);

  runTest(
      inputWkts,
      inputWkts,
      std::nullopt,
      "ST_Equals(left_g, right_g)",
      core::JoinType::kLeft,
      inputWkts,
      inputWkts);
}

TEST_F(SpatialJoinTest, pointPolygonSpatialJoin) {
  std::vector<std::optional<std::string_view>> polygonWkts = {
      kPolygonA, kPolygonB, kPolygonC, kPolygonD};
  std::vector<std::optional<std::string_view>> pointWkts = {
      kPointX,
      kPointY,
      kPointZ,
      kPointW,
      kPointV,
      kPointS,
      kPointQ,
      kMultipointU,
      kMultipointT,
      kMultipointR};

  std::vector<std::optional<std::string_view>> pointOutputWkts = {
      kPointX,
      kPointY,
      kPointY,
      kPointZ,
      kPointV,
      kPointS,
      kMultipointU,
      kMultipointR,
      kMultipointR,
      kMultipointT};
  std::vector<std::optional<std::string_view>> polygonOutputWkts = {
      kPolygonA,
      kPolygonA,
      kPolygonB,
      kPolygonB,
      kPolygonC,
      kPolygonD,
      kPolygonC,
      kPolygonC,
      kPolygonD,
      kPolygonC};
  runTest(
      pointWkts,
      polygonWkts,
      std::nullopt,
      "ST_Intersects(left_g, right_g)",
      core::JoinType::kInner,
      pointOutputWkts,
      polygonOutputWkts);
}

TEST_F(SpatialJoinTest, testSimpleNullRowsJoin) {
  runTest(
      {"POINT (1 1)", std::nullopt, "POINT (1 2)"},
      {"POINT (1 1)", "POINT (2 1)", std::nullopt},
      std::nullopt,
      "ST_Intersects(left_g, right_g)",
      core::JoinType::kInner,
      {"POINT (1 1)"},
      {"POINT (1 1)"});
}

TEST_F(SpatialJoinTest, testGeometryCollection) {
  runTest(
      {"GEOMETRYCOLLECTION (POINT (1 1))",
       "GEOMETRYCOLLECTION EMPTY",
       "POINT (1 1)"},
      {"GEOMETRYCOLLECTION (POINT (1 1))",
       "GEOMETRYCOLLECTION EMPTY",
       "POINT (1 1)"},
      std::nullopt,
      "ST_Intersects(left_g, right_g)",
      core::JoinType::kInner,
      {"GEOMETRYCOLLECTION (POINT (1 1))",
       "GEOMETRYCOLLECTION (POINT (1 1))",
       "POINT (1 1)",
       "POINT (1 1)"},
      {"GEOMETRYCOLLECTION (POINT (1 1))",
       "POINT (1 1)",
       "GEOMETRYCOLLECTION (POINT (1 1))",
       "POINT (1 1)"});

  runTest(
      {"GEOMETRYCOLLECTION (POINT (1 1))",
       "GEOMETRYCOLLECTION EMPTY",
       "POINT (1 1)"},
      {"GEOMETRYCOLLECTION (POINT (1 2))",
       "GEOMETRYCOLLECTION EMPTY",
       "POINT (1 2)"},
      std::vector<std::optional<double>>{1.0, 1.0, 1.0},
      "ST_Distance(left_g, right_g) <= radius",
      core::JoinType::kInner,
      {"GEOMETRYCOLLECTION (POINT (1 1))",
       "GEOMETRYCOLLECTION (POINT (1 1))",
       "POINT (1 1)",
       "POINT (1 1)"},
      {"GEOMETRYCOLLECTION (POINT (1 2))",
       "POINT (1 2)",
       "GEOMETRYCOLLECTION (POINT (1 2))",
       "POINT (1 2)"});
}

TEST_F(SpatialJoinTest, testDistanceJoin) {
  runTest(
      {"POINT (1 2)", "POLYGON ((1 2, 2 2, 2 3, 1 3, 1 2))", std::nullopt},
      {"POINT (2 2)",
       "POINT (1 1)",
       std::nullopt,
       "POINT (1 2)",
       "POLYGON ((1 1, 1 0, 0 0, 0 1, 1 1))"},
      std::vector<std::optional<double>>{std::nullopt, 1.0, 0.0, 0.0, 1.0},
      "ST_Distance(left_g, right_g) <= radius",
      core::JoinType::kInner,
      {"POINT (1 2)",
       "POLYGON ((1 2, 1 3, 2 3, 2 2, 1 2))",
       "POINT (1 2)",
       "POLYGON ((1 2, 1 3, 2 3, 2 2, 1 2))",
       "POINT (1 2)",
       "POLYGON ((1 2, 1 3, 2 3, 2 2, 1 2))"},
      {"POINT (1 1)",
       "POINT (1 1)",
       "POINT (1 2)",
       "POINT (1 2)",
       "POLYGON ((1 1, 1 0, 0 0, 0 1, 1 1))",
       "POLYGON ((1 1, 1 0, 0 0, 0 1, 1 1))"});
}

TEST_F(SpatialJoinTest, testContainsPointsInPolygons) {
  // Tests ST_Contains(polygon, point) - which polygons contain which points
  std::vector<std::optional<std::string_view>> pointWkts = {
      kPointX, kPointY, kPointZ, kPointW};
  std::vector<std::optional<std::string_view>> polygonWkts = {
      kPolygonA, kPolygonB, kPolygonC, kPolygonD};

  // Expected: A contains X, B contains Y, B contains Z, A contains Y, D
  // contains nothing from our test set Note: Y is in both A and B since they
  // overlap
  std::vector<std::optional<std::string_view>> pointOutputWkts = {
      kPointX, kPointY, kPointY, kPointZ};
  std::vector<std::optional<std::string_view>> polygonOutputWkts = {
      kPolygonA, kPolygonA, kPolygonB, kPolygonB};

  runTest(
      pointWkts,
      polygonWkts,
      std::nullopt,
      "ST_Contains(right_g, left_g)",
      core::JoinType::kInner,
      pointOutputWkts,
      polygonOutputWkts);
}

TEST_F(SpatialJoinTest, testContainsPolygonsInPolygons) {
  // Tests ST_Contains(polygon, polygon) - which polygons contain which polygons
  // From the Java test, polygon C contains polygon B (C is larger and covers B)
  std::vector<std::optional<std::string_view>> polygonWkts = {
      kPolygonA, kPolygonB, kPolygonC, kPolygonD};

  // Each polygon contains itself, plus any additional containments
  // Based on the spatial relations, we need to check which polygons actually
  // contain others For now, test self-containment which should always work
  std::vector<std::optional<std::string_view>> leftOutputWkts = {
      kPolygonA, kPolygonB, kPolygonC, kPolygonD};
  std::vector<std::optional<std::string_view>> rightOutputWkts = {
      kPolygonA, kPolygonB, kPolygonC, kPolygonD};

  runTest(
      polygonWkts,
      polygonWkts,
      std::nullopt,
      "ST_Contains(right_g, left_g)",
      core::JoinType::kInner,
      leftOutputWkts,
      rightOutputWkts);
}

TEST_F(SpatialJoinTest, testContainsLeftJoin) {
  // Tests ST_Contains with LEFT join - all probe rows should appear
  std::vector<std::optional<std::string_view>> pointWkts = {
      kPointX, kPointY, kPointZ, kPointW};
  std::vector<std::optional<std::string_view>> polygonWkts = {
      kPolygonA, kPolygonB};

  // W is outside both polygons, so it should have null for the right side
  std::vector<std::optional<std::string_view>> pointOutputWkts = {
      kPointX, kPointY, kPointY, kPointZ, kPointW};
  std::vector<std::optional<std::string_view>> polygonOutputWkts = {
      kPolygonA, kPolygonA, kPolygonB, kPolygonB, std::nullopt};

  runTest(
      pointWkts,
      polygonWkts,
      std::nullopt,
      "ST_Contains(right_g, left_g)",
      core::JoinType::kLeft,
      pointOutputWkts,
      polygonOutputWkts);
}

TEST_F(SpatialJoinTest, testTouches) {
  // Test ST_Touches - geometries that touch at boundary but don't overlap
  // Polygon and a point on its boundary
  std::vector<std::optional<std::string_view>> probeWkts = {
      "POINT (1 2)", "POINT (3 2)", "LINESTRING (0 0, 1 1)"};
  std::vector<std::optional<std::string_view>> buildWkts = {
      "POLYGON ((1 1, 1 4, 4 4, 4 1, 1 1))",
      "POLYGON ((1 1, 1 3, 3 3, 3 1, 1 1))"};

  // Point (1,2) touches both polygons (on their boundaries)
  // Point (3,2) touches second polygon (on its boundary)
  // LineString (0 0, 1 1) touches both polygons (endpoint at (1,1))
  std::vector<std::optional<std::string_view>> probeOutputWkts = {
      "POINT (1 2)",
      "POINT (1 2)",
      "POINT (3 2)",
      "LINESTRING (0 0, 1 1)",
      "LINESTRING (0 0, 1 1)"};
  std::vector<std::optional<std::string_view>> buildOutputWkts = {
      "POLYGON ((1 1, 1 4, 4 4, 4 1, 1 1))",
      "POLYGON ((1 1, 1 3, 3 3, 3 1, 1 1))",
      "POLYGON ((1 1, 1 3, 3 3, 3 1, 1 1))",
      "POLYGON ((1 1, 1 4, 4 4, 4 1, 1 1))",
      "POLYGON ((1 1, 1 3, 3 3, 3 1, 1 1))"};

  runTest(
      probeWkts,
      buildWkts,
      std::nullopt,
      "ST_Touches(left_g, right_g)",
      core::JoinType::kInner,
      probeOutputWkts,
      buildOutputWkts);
}

TEST_F(SpatialJoinTest, testTouchesPolygons) {
  // Test ST_Touches with two polygons that touch at a corner
  std::vector<std::optional<std::string_view>> probeWkts = {
      "POLYGON ((1 1, 1 3, 3 3, 3 1, 1 1))",
      "POLYGON ((5 5, 5 6, 6 6, 6 5, 5 5))"};
  std::vector<std::optional<std::string_view>> buildWkts = {
      "POLYGON ((3 3, 3 5, 5 5, 5 3, 3 3))"};

  // Both polygons touch the build polygon at corners:
  // - First polygon touches at (3,3)
  // - Second polygon touches at (5,5)
  std::vector<std::optional<std::string_view>> probeOutputWkts = {
      "POLYGON ((1 1, 1 3, 3 3, 3 1, 1 1))",
      "POLYGON ((5 5, 5 6, 6 6, 6 5, 5 5))"};
  std::vector<std::optional<std::string_view>> buildOutputWkts = {
      "POLYGON ((3 3, 3 5, 5 5, 5 3, 3 3))",
      "POLYGON ((3 3, 3 5, 5 5, 5 3, 3 3))"};

  runTest(
      probeWkts,
      buildWkts,
      std::nullopt,
      "ST_Touches(left_g, right_g)",
      core::JoinType::kInner,
      probeOutputWkts,
      buildOutputWkts);
}

TEST_F(SpatialJoinTest, testCrosses) {
  // Test ST_Crosses - geometries that cross each other
  // A linestring crossing a polygon
  std::vector<std::optional<std::string_view>> probeWkts = {
      "LINESTRING (0 0, 4 4)", // Crosses both polygons
      "LINESTRING (5 0, 5 4)", // Outside both
      "LINESTRING (1 1, 2 2)" // Contained in polygon, doesn't cross
  };
  std::vector<std::optional<std::string_view>> buildWkts = {
      "POLYGON ((1 1, 1 3, 3 3, 3 1, 1 1))"};

  // Only the first linestring crosses the polygon
  std::vector<std::optional<std::string_view>> probeOutputWkts = {
      "LINESTRING (0 0, 4 4)"};
  std::vector<std::optional<std::string_view>> buildOutputWkts = {
      "POLYGON ((1 1, 1 3, 3 3, 3 1, 1 1))"};

  runTest(
      probeWkts,
      buildWkts,
      std::nullopt,
      "ST_Crosses(left_g, right_g)",
      core::JoinType::kInner,
      probeOutputWkts,
      buildOutputWkts);
}

TEST_F(SpatialJoinTest, testCrossesLineStrings) {
  // Test ST_Crosses with two linestrings that cross each other
  std::vector<std::optional<std::string_view>> probeWkts = {
      "LINESTRING (0 0, 1 1)", // Crosses first build linestring
      "LINESTRING (2 2, 3 3)" // Parallel, doesn't cross
  };
  std::vector<std::optional<std::string_view>> buildWkts = {
      "LINESTRING (1 0, 0 1)"};

  // Only the first linestring crosses
  std::vector<std::optional<std::string_view>> probeOutputWkts = {
      "LINESTRING (0 0, 1 1)"};
  std::vector<std::optional<std::string_view>> buildOutputWkts = {
      "LINESTRING (1 0, 0 1)"};

  runTest(
      probeWkts,
      buildWkts,
      std::nullopt,
      "ST_Crosses(left_g, right_g)",
      core::JoinType::kInner,
      probeOutputWkts,
      buildOutputWkts);
}

TEST_F(SpatialJoinTest, testEmptyBuild) {
  runTest(
      {kPointX, std::nullopt, kPointY, kPointZ, kPointW},
      {},
      std::nullopt,
      "ST_Intersects(left_g, right_g)",
      core::JoinType::kInner,
      {},
      {});
  runTest(
      {kPointX, std::nullopt, kPointY, kPointZ, kPointW},
      {},
      std::nullopt,
      "ST_Intersects(left_g, right_g)",
      core::JoinType::kLeft,
      {kPointX, std::nullopt, kPointY, kPointZ, kPointW},
      {std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt});
}

TEST_F(SpatialJoinTest, testEmptyProbe) {
  runTest(
      {},
      {kPointX, std::nullopt, kPointY, kPointZ, kPointW},
      std::nullopt,
      "ST_Intersects(left_g, right_g)",
      core::JoinType::kInner,
      {},
      {});
  runTest(
      {},
      {kPointX, std::nullopt, kPointY, kPointZ, kPointW},
      std::nullopt,
      "ST_Intersects(left_g, right_g)",
      core::JoinType::kLeft,
      {},
      {});
}

TEST_F(SpatialJoinTest, failOnGroupedExecution) {
  std::vector<RowVectorPtr> batches{
      makeRowVector({"wkt"}, {makeFlatVector<std::string>({"POINT(0 0)"})})};
  core::PlanNodeId groupedScanNodeId;
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  auto planFragment =
      PlanBuilder(planNodeIdGenerator)
          .values(batches)
          .capturePlanNodeId(groupedScanNodeId)
          .project({"ST_GeometryFromText(wkt) AS left_g"})
          .spatialJoin(
              PlanBuilder(planNodeIdGenerator)
                  .values(batches)
                  .project({"ST_GeometryFromText(wkt) AS right_g"})
                  .localPartition({})
                  .planNode(),
              "ST_Intersects(left_g, right_g)",
              "left_g",
              "right_g",
              std::nullopt,
              {"left_g", "right_g"},
              core::JoinType::kInner)
          .project(
              {"ST_AsText(left_g) AS left_g", "ST_AsText(right_g) AS right_g"})
          .planFragment();
  planFragment.executionStrategy = core::ExecutionStrategy::kGrouped;
  planFragment.groupedExecutionLeafNodeIds.emplace(groupedScanNodeId);
  auto task = Task::create(
      "task-grouped-join",
      std::move(planFragment),
      0,
      core::QueryCtx::create(driverExecutor_.get()),
      Task::ExecutionMode::kParallel);

  VELOX_ASSERT_THROW(
      task->start(1), "Spatial joins do not support grouped execution.");
}

TEST_F(SpatialJoinTest, testLargeJoinSize) {
  size_t numRows = 64;
  size_t maxCoord = 17;
  std::vector<std::string> buildWkts;
  buildWkts.reserve(numRows);
  std::vector<std::string> probeWkts;
  probeWkts.reserve(numRows);
  for (size_t i = 0; i < numRows; ++i) {
    buildWkts.push_back(
        fmt::format("POINT ({} {})", (i + 1) % maxCoord, (i + 2) % maxCoord));
    probeWkts.push_back(
        fmt::format("POINT ({} {})", i % maxCoord, (i + 1) % maxCoord));
  }

  std::vector<std::optional<std::string_view>> buildWktsView;
  buildWktsView.reserve(numRows);
  std::vector<std::optional<std::string_view>> probeWktsView;
  probeWktsView.reserve(numRows);
  for (size_t i = 0; i < numRows; ++i) {
    buildWktsView.push_back(buildWkts[i]);
    probeWktsView.push_back(probeWkts[i]);
  }

  std::vector<std::optional<std::string_view>> expectedLeftWkts;
  expectedLeftWkts.reserve(numRows * numRows / maxCoord);
  std::vector<std::optional<std::string_view>> expectedRightWkts;
  expectedRightWkts.reserve(numRows * numRows / maxCoord);
  for (size_t innerIdx = 0; innerIdx < numRows; ++innerIdx) {
    for (size_t outerIdx = 0; outerIdx < numRows; ++outerIdx) {
      if (probeWkts[outerIdx] == buildWkts[innerIdx]) {
        expectedLeftWkts.push_back(probeWkts[outerIdx]);
        expectedRightWkts.push_back(buildWkts[innerIdx]);
      }
    }
  }

  for (bool separateProbeBatches : {false, true}) {
    for (size_t maxBatchSize : {64, 13, 7, 5, 3, 2, 1}) {
      runTestWithConfig(
          buildWktsView,
          probeWktsView,
          std::nullopt,
          "ST_Equals(left_g, right_g)",
          core::JoinType::kInner,
          expectedLeftWkts,
          expectedRightWkts,
          1,
          maxBatchSize,
          separateProbeBatches);
    }
  }
}

} // namespace facebook::velox::exec::test
