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

using facebook::velox::functions::test::FunctionBaseTest;

class S2FunctionsTest : public FunctionBaseTest {
 protected:
  void testS2CellIdParent(
      const std::optional<std::string>& cellId,
      const std::optional<int32_t>& level,
      const std::optional<int64_t>& expected) {
    auto result =
        evaluateOnce<int64_t>("s2_cell_id_parent(c0, c1)", cellId, level);
    if (expected.has_value()) {
      ASSERT_TRUE(result.has_value());
      EXPECT_EQ(expected.value(), result.value());
    } else {
      EXPECT_FALSE(result.has_value());
    }
  }

  void testS2CellIdAreaSqKm(
      const std::optional<std::string>& cellId,
      const std::optional<double>& expected) {
    auto result = evaluateOnce<double>("s2_cell_id_area_sq_km(c0)", cellId);
    if (expected.has_value()) {
      ASSERT_TRUE(result.has_value());
      EXPECT_NEAR(expected.value(), result.value(), 0.000001);
    } else {
      EXPECT_FALSE(result.has_value());
    }
  }

  void testS2CellTokenParent(
      const std::optional<std::string>& cellToken,
      const std::optional<int32_t>& level,
      const std::optional<std::string>& expected) {
    auto result = evaluateOnce<std::string>(
        "s2_cell_token_parent(c0, c1)", cellToken, level);
    if (expected.has_value()) {
      ASSERT_TRUE(result.has_value());
      EXPECT_EQ(expected.value(), result.value());
    } else {
      EXPECT_FALSE(result.has_value());
    }
  }
};

TEST_F(S2FunctionsTest, s2CellIdParent) {
  // Test getting parent at different levels
  // Note: S2 cell IDs are stored as int64_t (reinterpreted from uint64_t)
  testS2CellIdParent("9260949539409362944", 14, -9185794530463449088LL);

  // Test with the same level (should return the same cell ID)
  testS2CellIdParent("9260949543246102528", 14, -9185794530463449088LL);

  // Test with a higher level (should return the same cell ID)
  testS2CellIdParent("9260949543246102528", 15, -9185794530463449088LL);

  // Test with null inputs
  testS2CellIdParent(std::nullopt, 2, std::nullopt);
  testS2CellIdParent("9260949543246102528", std::nullopt, std::nullopt);
  testS2CellIdParent(std::nullopt, std::nullopt, std::nullopt);

  // Test with invalid level
  VELOX_ASSERT_USER_THROW(
      testS2CellIdParent("9260949543246102528", -1, std::nullopt),
      "S2_CELL_ID_PARENT: Expected level -1 to be in [0,30] range");
  VELOX_ASSERT_USER_THROW(
      testS2CellIdParent("9260949543246102528", 31, std::nullopt),
      "S2_CELL_ID_PARENT: Expected level 31 to be in [0,30] range");

  // Test with invalid cell ID
  VELOX_ASSERT_USER_THROW(
      testS2CellIdParent("invalid", 15, std::nullopt),
      "S2_CELL_ID_PARENT: Invalid cell ID: invalid");
}

TEST_F(S2FunctionsTest, s2CellIdAreaSqKm) {
  // Test area calculation for different cell IDs
  // Level 0 cell (1/6 of Earth's surface)
  testS2CellIdAreaSqKm("5764607523034234880", 85011012.18633142);

  // Level 14 cell
  testS2CellIdAreaSqKm("9260949543246102528", 0.263649755299609);

  // Test with null input
  testS2CellIdAreaSqKm(std::nullopt, std::nullopt);

  // Test with invalid cell ID
  VELOX_ASSERT_USER_THROW(
      testS2CellIdAreaSqKm("invalid", std::nullopt),
      "S2_CELL_AREA_SQ_KM: Invalid cell ID: invalid");
}

TEST_F(S2FunctionsTest, s2CellTokenParent) {
  // Test getting parent at different levels
  testS2CellTokenParent("8085808a1b5", 14, "8085808b");

  // Test with the same level (should return the same token)
  testS2CellTokenParent("8085808b", 14, "8085808b");

  // Test with a higher level (should return the same token)
  testS2CellTokenParent("8085808b", 15, "8085808b");

  // Test with null inputs
  testS2CellTokenParent(std::nullopt, 2, std::nullopt);
  testS2CellTokenParent("8085808b", std::nullopt, std::nullopt);
  testS2CellTokenParent(std::nullopt, std::nullopt, std::nullopt);

  // Test with invalid level
  VELOX_ASSERT_USER_THROW(
      testS2CellTokenParent("8085808b", -1, std::nullopt),
      "S2_CELL_TOKEN_PARENT: Expected level -1 to be in [0,30] range");
  VELOX_ASSERT_USER_THROW(
      testS2CellTokenParent("8085808b", 31, std::nullopt),
      "S2_CELL_TOKEN_PARENT: Expected level 31 to be in [0,30] range");

  // Test with invalid cell token
  VELOX_ASSERT_USER_THROW(
      testS2CellTokenParent("invalid", 15, std::nullopt),
      "S2_CELL_TOKEN_PARENT: Invalid cell token: invalid");
}
