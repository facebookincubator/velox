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
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/utils/AggregationTestBase.h"

using namespace facebook::velox::exec;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::aggregate::test {

namespace {

class GeometryAggregateTest : public AggregationTestBase {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
  }

  void testGeometryAggregate(
      const std::vector<std::string>& inputWkts,
      std::string_view aggregation,
      std::optional<std::string> outputWkt) {
    auto data = makeRowVector({
        makeFlatVector(inputWkts),
    });

    std::string outputSql;
    if (outputWkt.has_value()) {
      outputSql = fmt::format("SELECT '{}'", outputWkt.value());
    } else {
      outputSql = "SELECT NULL";
    }

    testAggregations(
        [&](exec::test::PlanBuilder& builder) {
          builder.values({data}).project({"st_geometryfromtext(c0) AS geom"});
        },
        {},
        {fmt::format("{}(geom) as agg", aggregation)},
        {"st_astext(agg)"},
        [&](auto& builder) { return builder.assertResults(outputSql); });
  }
};

TEST_F(GeometryAggregateTest, convexHullNoRows) {
  testGeometryAggregate({}, "convex_hull_agg", std::nullopt);
}

TEST_F(GeometryAggregateTest, convexHullSingleEmpty) {
  testGeometryAggregate(
      {"POINT EMPTY"}, "convex_hull_agg", "GEOMETRYCOLLECTION EMPTY");
}

TEST_F(GeometryAggregateTest, convexHullSinglePoint) {
  testGeometryAggregate({"POINT (1 2)"}, "convex_hull_agg", "POINT (1 2)");
}

TEST_F(GeometryAggregateTest, convexHullMultiplePoints) {
  testGeometryAggregate(
      {"POINT (0 0)", "POINT (1 1)", "POINT (1 0)"},
      "convex_hull_agg",
      "POLYGON ((0 0, 1 1, 1 0, 0 0))");
}

TEST_F(GeometryAggregateTest, convexHullMultipleLinearPoints) {
  testGeometryAggregate(
      {"POINT (1 2)", "POINT (3 4)", "POINT (4 5)"},
      "convex_hull_agg",
      "LINESTRING (1 2, 4 5)");
}

TEST_F(GeometryAggregateTest, convexHullSinglePolygon) {
  testGeometryAggregate(
      {"POLYGON ((0 0, 1 1, 1 0, 0 0), (0.5 0.5, 0.6 0.6, 0.6 0.5, 0.5 0.5))"},
      "convex_hull_agg",
      "POLYGON ((0 0, 1 1, 1 0, 0 0))");
}

TEST_F(GeometryAggregateTest, convexHullLineString) {
  testGeometryAggregate(
      {"LINESTRING (0 0, 1 1, 2 0)"},
      "convex_hull_agg",
      "POLYGON ((0 0, 1 1, 2 0, 0 0))");
}

TEST_F(GeometryAggregateTest, convexHullMultipleLineStrings) {
  testGeometryAggregate(
      {"LINESTRING (0 0, 1 1)", "LINESTRING (1 0, 2 1)"},
      "convex_hull_agg",
      "POLYGON ((0 0, 1 1, 2 1, 1 0, 0 0))");
}

TEST_F(GeometryAggregateTest, convexHullMultiPoint) {
  testGeometryAggregate(
      {"MULTIPOINT (0 0, 1 1, 2 0)"},
      "convex_hull_agg",
      "POLYGON ((0 0, 1 1, 2 0, 0 0))");
}

TEST_F(GeometryAggregateTest, convexHullMultiLineString) {
  testGeometryAggregate(
      {"MULTILINESTRING ((0 0, 1 1), (1 0, 2 1))"},
      "convex_hull_agg",
      "POLYGON ((0 0, 1 1, 2 1, 1 0, 0 0))");
}

TEST_F(GeometryAggregateTest, convexHullMultiPolygon) {
  testGeometryAggregate(
      {"MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))"},
      "convex_hull_agg",
      "POLYGON ((0 0, 0 1, 2 3, 3 3, 3 2, 1 0, 0 0))");
}

TEST_F(GeometryAggregateTest, convexHullGeometryCollection) {
  testGeometryAggregate(
      {"GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (1 1, 2 1), POLYGON ((3 0, 4 0, 4 1, 3 1, 3 0)))"},
      "convex_hull_agg",
      "POLYGON ((0 0, 1 1, 4 1, 4 0, 0 0))");
}

TEST_F(GeometryAggregateTest, convexHullMixedGeometryTypes) {
  testGeometryAggregate(
      {"POINT (0 0)",
       "LINESTRING (1 1, 2 1)",
       "POLYGON ((3 0, 4 0, 4 1, 3 1, 3 0))"},
      "convex_hull_agg",
      "POLYGON ((0 0, 1 1, 4 1, 4 0, 0 0))");
}

TEST_F(GeometryAggregateTest, convexHullOverlappingPoints) {
  testGeometryAggregate(
      {"POINT (0 0)", "POINT (0 0)", "POINT (1 1)", "POINT (1 0)"},
      "convex_hull_agg",
      "POLYGON ((0 0, 1 1, 1 0, 0 0))");
}

TEST_F(GeometryAggregateTest, convexHullOverlappingPolygons) {
  testGeometryAggregate(
      {"POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))",
       "POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"},
      "convex_hull_agg",
      "POLYGON ((0 0, 0 2, 1 3, 3 3, 3 1, 2 0, 0 0))");
}

TEST_F(GeometryAggregateTest, convexHullNonOverlappingPolygons) {
  testGeometryAggregate(
      {"POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
       "POLYGON ((3 3, 4 3, 4 4, 3 4, 3 3))"},
      "convex_hull_agg",
      "POLYGON ((0 0, 0 1, 3 4, 4 4, 4 3, 1 0, 0 0))");
}

TEST_F(GeometryAggregateTest, convexHullDuplicateCoordinatesAcrossGeometries) {
  testGeometryAggregate(
      {"POINT (0 0)",
       "LINESTRING (0 0, 1 1)",
       "POLYGON ((0 0, 2 0, 1 1, 0 0))"},
      "convex_hull_agg",
      "POLYGON ((0 0, 1 1, 2 0, 0 0))");
}

TEST_F(
    GeometryAggregateTest,
    convexHullDuplicateCoordinatesMultipleGeometries) {
  testGeometryAggregate(
      {"POINT (1 1)",
       "POINT (1 1)",
       "LINESTRING (1 1, 2 2)",
       "POLYGON ((1 1, 3 1, 2 2, 1 1))"},
      "convex_hull_agg",
      "POLYGON ((1 1, 2 2, 3 1, 1 1))");
}

TEST_F(GeometryAggregateTest, convexHullMixedWithEmpty) {
  testGeometryAggregate(
      {"POINT EMPTY", "POINT (1 1)", "POINT (2 2)", "POINT (1 2)"},
      "convex_hull_agg",
      "POLYGON ((1 1, 1 2, 2 2, 1 1))");
}

TEST_F(GeometryAggregateTest, convexHullMultipleEmpty) {
  testGeometryAggregate(
      {"POINT EMPTY", "LINESTRING EMPTY", "POLYGON EMPTY"},
      "convex_hull_agg",
      "GEOMETRYCOLLECTION EMPTY");
}

TEST_F(GeometryAggregateTest, convexHullComplexGeometryCollection) {
  testGeometryAggregate(
      {"GEOMETRYCOLLECTION (POINT (0 0), MULTIPOINT (1 1, 2 2))",
       "GEOMETRYCOLLECTION (LINESTRING (3 3, 4 4), POLYGON ((5 5, 6 5, 6 6, 5 6, 5 5)))"},
      "convex_hull_agg",
      "POLYGON ((0 0, 5 6, 6 6, 6 5, 0 0))");
}

TEST_F(GeometryAggregateTest, convexHullOverlappingLineStrings) {
  testGeometryAggregate(
      {"LINESTRING (0 0, 2 2)",
       "LINESTRING (1 1, 3 3)",
       "LINESTRING (0 2, 2 0)"},
      "convex_hull_agg",
      "POLYGON ((0 0, 0 2, 3 3, 2 0, 0 0))");
}

TEST_F(GeometryAggregateTest, convexHullNonOverlappingLineStrings) {
  testGeometryAggregate(
      {"LINESTRING (0 0, 1 0)", "LINESTRING (5 5, 6 5)"},
      "convex_hull_agg",
      "POLYGON ((0 0, 5 5, 6 5, 1 0, 0 0))");
}

TEST_F(GeometryAggregateTest, convexHullAllSamePoint) {
  testGeometryAggregate(
      {"POINT (5 5)", "POINT (5 5)", "POINT (5 5)"},
      "convex_hull_agg",
      "POINT (5 5)");
}

TEST_F(GeometryAggregateTest, convexHullCollinearPoints) {
  testGeometryAggregate(
      {"POINT (0 0)", "POINT (1 1)", "POINT (2 2)", "POINT (3 3)"},
      "convex_hull_agg",
      "LINESTRING (0 0, 3 3)");
}

TEST_F(GeometryAggregateTest, convexHullMixedDimensionalGeometries) {
  testGeometryAggregate(
      {"POINT (0 0)",
       "LINESTRING (1 0, 2 0)",
       "POLYGON ((0 1, 1 2, 2 1, 0 1))"},
      "convex_hull_agg",
      "POLYGON ((0 0, 0 1, 1 2, 2 1, 2 0, 0 0))");
}

TEST_F(GeometryAggregateTest, convexHullNestedGeometryCollections) {
  testGeometryAggregate(
      {"GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (POINT (0 0), POINT (1 1)), POINT (2 2))",
       "POINT (0 2)"},
      "convex_hull_agg",
      "POLYGON ((0 0, 0 2, 2 2, 0 0))");
}

TEST_F(GeometryAggregateTest, convexHullLargeCoordinates) {
  testGeometryAggregate(
      {"POINT (1000000 2000000)",
       "POINT (3000000 4000000)",
       "POINT (1000000 4000000)"},
      "convex_hull_agg",
      "POLYGON ((1000000 2000000, 1000000 4000000, 3000000 4000000, 1000000 2000000))");
}

TEST_F(GeometryAggregateTest, convexHullNegativeCoordinates) {
  testGeometryAggregate(
      {"POINT (-1 -1)", "POINT (1 1)", "POINT (-1 1)", "POINT (1 -1)"},
      "convex_hull_agg",
      "POLYGON ((-1 -1, -1 1, 1 1, 1 -1, -1 -1))");
}

TEST_F(GeometryAggregateTest, convexHullDecimalCoordinates) {
  testGeometryAggregate(
      {"POINT (0.5 0.5)",
       "POINT (1.5 1.5)",
       "POINT (0.5 1.5)",
       "POINT (1.5 0.5)"},
      "convex_hull_agg",
      "POLYGON ((0.5 0.5, 0.5 1.5, 1.5 1.5, 1.5 0.5, 0.5 0.5))");
}

TEST_F(GeometryAggregateTest, geometryUnionNoRows) {
  testGeometryAggregate({}, "geometry_union_agg", std::nullopt);
}

TEST_F(GeometryAggregateTest, geometryUnionSingleEmpty) {
  testGeometryAggregate(
      {"POINT EMPTY"}, "geometry_union_agg", "GEOMETRYCOLLECTION EMPTY");
}

TEST_F(GeometryAggregateTest, geometryUnionMultipleEmpty) {
  testGeometryAggregate(
      {"POINT EMPTY", "LINESTRING EMPTY", "POLYGON EMPTY"},
      "geometry_union_agg",
      "GEOMETRYCOLLECTION EMPTY");
}

TEST_F(GeometryAggregateTest, geometryUnionSinglePoint) {
  testGeometryAggregate({"POINT (1 2)"}, "geometry_union_agg", "POINT (1 2)");
}

TEST_F(GeometryAggregateTest, geometryUnionRepeatedPoints) {
  testGeometryAggregate(
      {"POINT (1 2)", "POINT (1 2)", "POINT (1 2)"},
      "geometry_union_agg",
      "POINT (1 2)");
}

TEST_F(GeometryAggregateTest, geometryUnionMultiplePoints) {
  testGeometryAggregate(
      {"POINT (0 0)", "POINT (1 1)", "POINT (2 0)"},
      "geometry_union_agg",
      "MULTIPOINT (0 0, 1 1, 2 0)");
}

TEST_F(GeometryAggregateTest, geometryUnionRepeatedGeometries) {
  testGeometryAggregate(
      {"POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
       "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))"},
      "geometry_union_agg",
      "POLYGON ((0 1, 1 1, 1 0, 0 0, 0 1))");
}

TEST_F(GeometryAggregateTest, geometryUnionOverlappingPolygons) {
  testGeometryAggregate(
      {"POLYGON ((0 0, 0 2, 2 2, 2 0, 0 0))",
       "POLYGON ((1 1, 1 3, 3 3, 3 1, 1 1))"},
      "geometry_union_agg",
      "POLYGON ((0 2, 1 2, 1 3, 3 3, 3 1, 2 1, 2 0, 0 0, 0 2))"

  );
}

TEST_F(GeometryAggregateTest, geometryUnionMixedDimensionsPointLine) {
  testGeometryAggregate(
      {"POINT (0 0)", "LINESTRING (1 1, 2 2)"},
      "geometry_union_agg",
      "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (1 1, 2 2))");
}

TEST_F(GeometryAggregateTest, geometryUnionMixedDimensionsPointPolygon) {
  testGeometryAggregate(
      {"POINT (0 0)", "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))"},
      "geometry_union_agg",
      "GEOMETRYCOLLECTION (POINT (0 0), POLYGON ((1 1, 1 2, 2 2, 2 1, 1 1)))");
}

TEST_F(GeometryAggregateTest, geometryUnionMixedDimensionsLinePolygon) {
  testGeometryAggregate(
      {"LINESTRING (0 0, 1 1)", "POLYGON ((2 2, 3 2, 3 3, 2 3, 2 2))"},
      "geometry_union_agg",
      "GEOMETRYCOLLECTION (LINESTRING (0 0, 1 1), POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2)))");
}

TEST_F(GeometryAggregateTest, geometryUnionMixedDimensionsAll) {
  testGeometryAggregate(
      {"POINT (0 0)",
       "LINESTRING (1 1, 2 2)",
       "POLYGON ((3 3, 4 3, 4 4, 3 4, 3 3))"},
      "geometry_union_agg",
      "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (1 1, 2 2), POLYGON ((3 3, 3 4, 4 4, 4 3, 3 3)))");
}

TEST_F(GeometryAggregateTest, geometryUnionMixedWithEmpty) {
  testGeometryAggregate(
      {"POINT EMPTY", "POINT (1 1)", "POINT (2 2)"},
      "geometry_union_agg",
      "MULTIPOINT (1 1, 2 2)");
}

TEST_F(GeometryAggregateTest, geometryUnionLineStrings) {
  testGeometryAggregate(
      {"LINESTRING (0 0, 1 1)", "LINESTRING (1 1, 2 2)"},
      "geometry_union_agg",
      "MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))");
}

TEST_F(GeometryAggregateTest, geometryUnionAdjacentPolygons) {
  testGeometryAggregate(
      {"POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
       "POLYGON ((1 0, 2 0, 2 1, 1 1, 1 0))"},
      "geometry_union_agg",
      "POLYGON ((0 1, 1 1, 2 1, 2 0, 1 0, 0 0, 0 1))"

  );
}

TEST_F(GeometryAggregateTest, geometryUnionMultiPoint) {
  testGeometryAggregate(
      {"MULTIPOINT (0 0, 1 1)", "POINT (2 2)"},
      "geometry_union_agg",
      "MULTIPOINT (0 0, 1 1, 2 2)");
}

TEST_F(GeometryAggregateTest, geometryUnionMultiLineString) {
  testGeometryAggregate(
      {"MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))", "LINESTRING (4 4, 5 5)"},
      "geometry_union_agg",
      "MULTILINESTRING ((0 0, 1 1), (2 2, 3 3), (4 4, 5 5))");
}

TEST_F(GeometryAggregateTest, geometryUnionGeometryCollection) {
  testGeometryAggregate(
      {"GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (1 1, 2 2))"},
      "geometry_union_agg",
      "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (1 1, 2 2))");
}

TEST_F(GeometryAggregateTest, geometryUnionPointInsidePolygon) {
  testGeometryAggregate(
      {"POINT (0.5 0.5)", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"},
      "geometry_union_agg",
      "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))");
}

TEST_F(GeometryAggregateTest, geometryUnionLineInsidePolygon) {
  testGeometryAggregate(
      {"LINESTRING (0.2 0.5, 0.8 0.5)", "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"},
      "geometry_union_agg",
      "POLYGON ((0 1, 1 1, 1 0, 0 0, 0 1))"

  );
}

TEST_F(GeometryAggregateTest, geometryUnionNegativeCoordinates) {
  testGeometryAggregate(
      {"POINT (-1 -1)", "POINT (1 1)"},
      "geometry_union_agg",
      "MULTIPOINT (-1 -1, 1 1)");
}

TEST_F(GeometryAggregateTest, geometryUnionDecimalCoordinates) {
  testGeometryAggregate(
      {"POINT (0.5 0.5)", "POINT (1.5 1.5)"},
      "geometry_union_agg",
      "MULTIPOINT (0.5 0.5, 1.5 1.5)");
}

} // namespace
} // namespace facebook::velox::aggregate::test
