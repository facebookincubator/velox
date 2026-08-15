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
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"

namespace facebook::velox::functions {
namespace {

// A level-0 face cell covering approximately 1/6 of Earth's surface.
constexpr int64_t kLevel0CellId{
    static_cast<int64_t>(5'764'607'523'034'234'880ULL)};

// A level-14 cell. kLevel20CellId is a level-20 descendant of this cell, so
// s2_cell_parent(kLevel20CellId, 14) == kLevel14CellId.
constexpr int64_t kLevel14CellId{
    static_cast<int64_t>(9'260'949'543'246'102'528ULL)};

// A level-20 descendant of kLevel14CellId.
constexpr int64_t kLevel20CellId{
    static_cast<int64_t>(9'260'949'539'409'362'944ULL)};

// Hex tokens corresponding to the cell IDs above.
constexpr std::string_view kLevel14Token{"8085808b"};
constexpr std::string_view kLevel20Token{"8085808a1b5"};

class S2FunctionsTest : public test::FunctionBaseTest {
 protected:
  // Evaluates s2_cells at a fixed level and returns the cell IDs.
  std::vector<int64_t> cells(const std::string& wkt, int32_t level) {
    return evaluate(
               "s2_cells(ST_GeometryFromText(c0), c1)",
               makeRowVector({
                   makeFlatVector({wkt}),
                   makeFlatVector<int32_t>({level}),
               }))
        ->variantAt(0)
        .array<int64_t>();
  }

  // Evaluates dissolved s2_cells and returns the cell IDs.
  std::vector<int64_t> cells(
      const std::string& wkt,
      int32_t minLevel,
      int32_t maxLevel,
      int32_t maxCells) {
    return evaluate(
               "s2_cells(ST_GeometryFromText(c0), c1, c2, c3)",
               makeRowVector({
                   makeFlatVector({wkt}),
                   makeFlatVector<int32_t>({minLevel}),
                   makeFlatVector<int32_t>({maxLevel}),
                   makeFlatVector<int32_t>({maxCells}),
               }))
        ->variantAt(0)
        .array<int64_t>();
  }

  // Returns the number of cells for convenience assertions.
  size_t countCells(const std::string& wkt, int32_t level) {
    return cells(wkt, level).size();
  }

  size_t countCells(
      const std::string& wkt,
      int32_t minLevel,
      int32_t maxLevel,
      int32_t maxCells) {
    return cells(wkt, minLevel, maxLevel, maxCells).size();
  }
};

TEST_F(S2FunctionsTest, parent) {
  auto test = [&](int64_t cellId, int32_t level, int64_t expected) {
    auto result = evaluateOnce<int64_t>(
        "s2_cell_parent(c0, c1)", std::optional(cellId), std::optional(level));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(expected, result.value());
  };

  test(kLevel20CellId, 14, kLevel14CellId);

  // Same level returns same cell ID.
  test(kLevel14CellId, 14, kLevel14CellId);

  // Higher level returns same cell ID.
  test(kLevel14CellId, 15, kLevel14CellId);

  // Invalid level.
  auto testThrows = [&](int32_t level, std::string_view message) {
    VELOX_ASSERT_USER_THROW(
        evaluateOnce<int64_t>(
            "s2_cell_parent(c0, c1)",
            std::optional(kLevel14CellId),
            std::optional(level)),
        message);
  };
  testThrows(-1, "s2_cell_parent: Level must be in [0, 30] range, got -1");
  testThrows(31, "s2_cell_parent: Level must be in [0, 30] range, got 31");
}

TEST_F(S2FunctionsTest, areaSqKm) {
  auto test = [&](int64_t cellId, double expected, double tolerance) {
    auto result =
        evaluateOnce<double>("s2_cell_area_sq_km(c0)", std::optional(cellId));
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(expected, result.value(), tolerance);
  };

  // Level 0 cell (1/6 of Earth's surface).
  test(kLevel0CellId, 85011012.18633142, 0.000001);

  // Level 14 cell.
  test(kLevel14CellId, 0.263649755299609, 0.000001);
}

TEST_F(S2FunctionsTest, fromToken) {
  auto test = [&](std::string_view token, int64_t expected) {
    auto result = evaluateOnce<int64_t>(
        "s2_cell_from_token(c0)", std::optional(std::string(token)));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(expected, result.value());
  };

  test(kLevel20Token, kLevel20CellId);
  test(kLevel14Token, kLevel14CellId);

  auto testThrows = [&](std::string_view token, std::string_view message) {
    VELOX_ASSERT_USER_THROW(
        evaluateOnce<int64_t>(
            "s2_cell_from_token(c0)", std::optional(std::string(token))),
        message);
  };

  // Invalid token.
  testThrows("invalid", "s2_cell_from_token: Invalid cell token: invalid");

  // Empty token.
  testThrows("", "s2_cell_from_token: Empty cell token");
}

TEST_F(S2FunctionsTest, contains) {
  // Level 14 cell contains its level 20 descendant.
  auto contains = evaluateOnce<bool>(
      "s2_cell_contains(c0, c1)",
      std::optional(kLevel14CellId),
      std::optional(kLevel20CellId));
  ASSERT_TRUE(contains.has_value());
  EXPECT_TRUE(contains.value());

  // Reverse is false.
  auto notContains = evaluateOnce<bool>(
      "s2_cell_contains(c0, c1)",
      std::optional(kLevel20CellId),
      std::optional(kLevel14CellId));
  ASSERT_TRUE(notContains.has_value());
  EXPECT_FALSE(notContains.value());

  // A cell contains itself.
  auto containsSelf = evaluateOnce<bool>(
      "s2_cell_contains(c0, c1)",
      std::optional(kLevel14CellId),
      std::optional(kLevel14CellId));
  ASSERT_TRUE(containsSelf.has_value());
  EXPECT_TRUE(containsSelf.value());

  // Composes with s2_cell_from_token without casting.
  auto composedResult = evaluateOnce<bool>(
      "s2_cell_contains(s2_cell_from_token(c0), s2_cell_from_token(c1))",
      std::optional(std::string(kLevel14Token)),
      std::optional(std::string(kLevel20Token)));
  ASSERT_TRUE(composedResult.has_value());
  EXPECT_TRUE(composedResult.value());
}

TEST_F(S2FunctionsTest, level) {
  auto level0 =
      evaluateOnce<int32_t>("s2_cell_level(c0)", std::optional(kLevel0CellId));
  ASSERT_TRUE(level0.has_value());
  EXPECT_EQ(0, level0.value());

  auto level14 =
      evaluateOnce<int32_t>("s2_cell_level(c0)", std::optional(kLevel14CellId));
  ASSERT_TRUE(level14.has_value());
  EXPECT_EQ(14, level14.value());
}

TEST_F(S2FunctionsTest, toToken) {
  auto test = [&](int64_t cellId, std::string_view expected) {
    auto result = evaluateOnce<std::string>(
        "s2_cell_to_token(c0)", std::optional(cellId));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(expected, result.value());
  };

  test(kLevel20CellId, kLevel20Token);
  test(kLevel14CellId, kLevel14Token);
}

TEST_F(S2FunctionsTest, cells) {
  // A point produces exactly one cell.
  EXPECT_EQ(1, countCells("POINT (0 0)", 10));

  // A polygon produces multiple cells.
  EXPECT_GT(countCells("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 10), 1);

  // Empty geometries produce an empty array.
  EXPECT_EQ(0, countCells("GEOMETRYCOLLECTION EMPTY", 10));
  EXPECT_EQ(0, countCells("POLYGON EMPTY", 10));

  // Non-empty GeometryCollection produces cells for each sub-geometry.
  EXPECT_GT(countCells("GEOMETRYCOLLECTION (POINT (0 0), POINT (1 1))", 10), 0);

  // LineString produces multiple cells.
  EXPECT_GT(countCells("LINESTRING (0 0, 1 1)", 10), 1);

  // MultiPoint produces one cell per point.
  EXPECT_EQ(3, countCells("MULTIPOINT (0 0, 10 10, 20 20)", 10));

  // Round-trip: level-10 cell contains level-20 cell for same point.
  auto level10Cells = cells("POINT (40.7128 -74.006)", 10);
  auto level20Cells = cells("POINT (40.7128 -74.006)", 20);
  ASSERT_EQ(1, level10Cells.size());
  ASSERT_EQ(1, level20Cells.size());
  auto containsResult = evaluateOnce<bool>(
      "s2_cell_contains(c0, c1)",
      std::optional(level10Cells[0]),
      std::optional(level20Cells[0]));
  ASSERT_TRUE(containsResult.has_value());
  EXPECT_TRUE(containsResult.value());

  // Level 0 and 30 edge cases.
  for (int32_t level : {0, 30}) {
    EXPECT_EQ(1, countCells("POINT (40.7128 -74.006)", level));
  }

  // Invalid levels.
  for (int32_t level : {-1, 31}) {
    VELOX_ASSERT_USER_THROW(
        cells("POINT (0 0)", level),
        "s2_cells: Level must be in [0, 30] range");
  }
}

TEST_F(S2FunctionsTest, cellsDissolved) {
  // Dissolved covering should produce fewer cells than fixed-level.
  auto dissolvedCells =
      cells("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 4, 14, 100);
  EXPECT_GT(dissolvedCells.size(), 0);
  EXPECT_LE(dissolvedCells.size(), 100);
  EXPECT_LT(
      dissolvedCells.size(),
      countCells("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", 14));

  // All cells should have levels within [min_level, max_level].
  for (auto cellId : dissolvedCells) {
    auto cellLevel =
        evaluateOnce<int32_t>("s2_cell_level(c0)", std::optional(cellId));
    ASSERT_TRUE(cellLevel.has_value());
    EXPECT_GE(cellLevel.value(), 4);
    EXPECT_LE(cellLevel.value(), 14);
  }

  // Point produces exactly 1 cell.
  EXPECT_EQ(1, countCells("POINT (0 0)", 0, 20, 100));

  // Empty geometry produces an empty array.
  EXPECT_EQ(0, countCells("GEOMETRYCOLLECTION EMPTY", 0, 20, 100));

  // min_level > max_level.
  VELOX_ASSERT_USER_THROW(
      cells("POINT (0 0)", 15, 10, 100),
      "s2_cells: min_level (15) must be <= max_level (10)");

  // max_cells < 1.
  VELOX_ASSERT_USER_THROW(
      cells("POINT (0 0)", 0, 20, 0), "s2_cells: max_cells must be >= 1");
}

} // namespace
} // namespace facebook::velox::functions
