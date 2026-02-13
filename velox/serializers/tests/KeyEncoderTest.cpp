/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/serializers/KeyEncoder.h"

#include <fmt/ranges.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <algorithm>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

namespace facebook::velox::serializer::test {
namespace {

class KeyEncoderTest : public velox::exec::test::OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
  }

  static int lexicographicalCompare(
      const std::string& key1,
      const std::string& key2) {
    const auto begin1 = reinterpret_cast<const unsigned char*>(key1.data());
    const auto end1 = begin1 + key1.size();
    const auto begin2 = reinterpret_cast<const unsigned char*>(key2.data());
    const auto end2 = begin2 + key2.size();
    bool lessThan = std::lexicographical_compare(begin1, end1, begin2, end2);

    bool equal = std::equal(begin1, end1, begin2, end2);

    return lessThan ? -1 : (equal ? 0 : 1);
  }

  // Helper method to test a specific sort order
  void encodeTestWithSortOrder(
      const std::vector<velox::RowVectorPtr>& inputs,
      const std::vector<std::string>& sortingKeys,
      const velox::core::SortOrder& sortOrder) {
    ASSERT_FALSE(inputs.empty());
    const auto inputType = asRowType(inputs[0]->type());

    // Build key column types string
    std::vector<std::string> keyColumnTypes;
    keyColumnTypes.reserve(sortingKeys.size());
    for (const auto& key : sortingKeys) {
      auto child = inputType->findChild(key);
      if (child) {
        keyColumnTypes.push_back(child->toString());
      }
    }

    SCOPED_TRACE(
        fmt::format(
            "Sort order: {}, nullsFirst: {}, Key columns: [{}]",
            sortOrder.isAscending() ? "ASC" : "DESC",
            sortOrder.isNullsFirst(),
            folly::join(", ", keyColumnTypes)));

    // Create KeyEncoder - always create a new one for each test
    auto keyEncoder = KeyEncoder::create(
        sortingKeys,
        inputType,
        std::vector<velox::core::SortOrder>{sortingKeys.size(), sortOrder},
        pool_.get());

    for (const auto& input : inputs) {
      // Encode the keys
      std::vector<char> buffer;
      std::vector<std::string_view> encodedKeys;
      keyEncoder->encode(input, encodedKeys, [&buffer](size_t size) -> void* {
        buffer.resize(size);
        return buffer.data();
      });

      // Sort indices based on encoded keys (lexicographic comparison)
      std::vector<velox::vector_size_t> encodedIndices(input->size());
      std::iota(encodedIndices.begin(), encodedIndices.end(), 0);
      std::sort(
          encodedIndices.begin(),
          encodedIndices.end(),
          [&encodedKeys](velox::vector_size_t a, velox::vector_size_t b) {
            return lexicographicalCompare(
                       std::string(encodedKeys[a]),
                       std::string(encodedKeys[b])) < 0;
          });

      // Build orderBy specification strings for all sorting keys
      std::vector<std::string> orderBySpecs;
      orderBySpecs.reserve(sortingKeys.size());
      for (const auto& key : sortingKeys) {
        std::string spec = key;
        if (!sortOrder.isAscending()) {
          spec += " DESC";
        }
        if (sortOrder.isNullsFirst()) {
          spec += " NULLS FIRST";
        } else {
          spec += " NULLS LAST";
        }
        orderBySpecs.push_back(spec);
      }

      // Use AssertQueryBuilder to create Velox OrderBy plan
      velox::exec::test::AssertQueryBuilder queryBuilder{
          velox::exec::test::PlanBuilder()
              .values({input})
              .orderBy(orderBySpecs, /*isPartial=*/false)
              .planNode(),
          duckDbQueryRunner_};

      // Execute the plan and get results
      auto veloxResult = queryBuilder.copyResults(pool_.get());

      // Compare results - reorder input based on encoded key sort order
      auto encodedResult = velox::BaseVector::create(
          inputType, encodedIndices.size(), pool_.get());
      for (size_t i = 0; i < encodedIndices.size(); ++i) {
        encodedResult->copy(input.get(), i, encodedIndices[i], 1);
      }

      // Verify both results match
      velox::test::assertEqualVectors(encodedResult, veloxResult);
    }
  }

  // Test method to validate KeyEncoder encoding against Velox OrderBy operator
  // Tests all four sort orders: AscNullsFirst, AscNullsLast, DescNullsFirst,
  // DescNullsLast
  void encodeTest(
      const std::vector<velox::RowVectorPtr>& inputs,
      const std::vector<std::string>& sortingKeys) {
    encodeTestWithSortOrder(inputs, sortingKeys, velox::core::kAscNullsFirst);
    encodeTestWithSortOrder(inputs, sortingKeys, velox::core::kAscNullsLast);
    encodeTestWithSortOrder(inputs, sortingKeys, velox::core::kDescNullsFirst);
    encodeTestWithSortOrder(inputs, sortingKeys, velox::core::kDescNullsFirst);
  }

  struct EncodeIndexBoundsTestCase {
    std::vector<std::string> indexColumns;
    std::optional<IndexBound> lowerBound;
    std::optional<IndexBound> upperBound;
    std::optional<velox::RowVectorPtr> expectedLowerBound;
    std::optional<velox::RowVectorPtr> expectedUpperBound;
    velox::core::SortOrder sortOrder = velox::core::kAscNullsFirst;
    bool expectedFailure = false;

    std::string debugString() const {
      auto vectorToString = [](const velox::VectorPtr& vec) -> std::string {
        if (vec->isNullAt(0)) {
          return "null";
        }

        const auto typeKind = vec->typeKind();
        switch (typeKind) {
          case velox::TypeKind::BIGINT: {
            const auto* flatVec = vec->as<velox::FlatVector<int64_t>>();
            return std::to_string(flatVec->valueAt(0));
          }
          case velox::TypeKind::INTEGER: {
            const auto* flatVec = vec->as<velox::FlatVector<int32_t>>();
            return std::to_string(flatVec->valueAt(0));
          }
          case velox::TypeKind::SMALLINT: {
            const auto* flatVec = vec->as<velox::FlatVector<int16_t>>();
            return std::to_string(flatVec->valueAt(0));
          }
          case velox::TypeKind::TINYINT: {
            const auto* flatVec = vec->as<velox::FlatVector<int8_t>>();
            return std::to_string(static_cast<int>(flatVec->valueAt(0)));
          }
          case velox::TypeKind::DOUBLE: {
            const auto* flatVec = vec->as<velox::FlatVector<double>>();
            return std::to_string(flatVec->valueAt(0));
          }
          case velox::TypeKind::REAL: {
            const auto* flatVec = vec->as<velox::FlatVector<float>>();
            return std::to_string(flatVec->valueAt(0));
          }
          case velox::TypeKind::BOOLEAN: {
            const auto* flatVec = vec->as<velox::FlatVector<bool>>();
            return flatVec->valueAt(0) ? "true" : "false";
          }
          case velox::TypeKind::VARCHAR:
          case velox::TypeKind::VARBINARY: {
            const auto* flatVec =
                vec->as<velox::FlatVector<velox::StringView>>();
            return flatVec->valueAt(0).str();
          }
          case velox::TypeKind::TIMESTAMP: {
            const auto* flatVec =
                vec->as<velox::FlatVector<velox::Timestamp>>();
            return flatVec->valueAt(0).toString();
          }
          default:
            return fmt::format(
                "<unknown type: {}>", static_cast<int>(typeKind));
        }
      };

      std::vector<std::string> parts;

      parts.push_back(
          fmt::format("columns=[{}]", folly::join(", ", indexColumns)));

      // Add sort order information
      std::string soStr = sortOrder.isAscending() ? "ASC" : "DESC";
      soStr += sortOrder.isNullsFirst() ? "_NULLS_FIRST" : "_NULLS_LAST";
      parts.push_back(fmt::format("sortOrder={}", soStr));

      if (lowerBound.has_value()) {
        const auto& bound = lowerBound.value();
        std::vector<std::string> values;
        values.reserve(bound.bound->childrenSize());
        for (size_t i = 0; i < bound.bound->childrenSize(); ++i) {
          values.emplace_back(vectorToString(bound.bound->childAt(i)));
        }
        parts.push_back(
            fmt::format(
                "lower={}[{}]",
                bound.inclusive ? ">=" : ">",
                folly::join(", ", values)));
      } else {
        parts.push_back("lower=null");
      }

      if (upperBound.has_value()) {
        const auto& bound = upperBound.value();
        std::vector<std::string> values;
        for (size_t i = 0; i < bound.bound->childrenSize(); ++i) {
          values.push_back(vectorToString(bound.bound->childAt(i)));
        }
        parts.push_back(
            fmt::format(
                "upper={}[{}]",
                bound.inclusive ? "<=" : "<",
                folly::join(", ", values)));
      } else {
        parts.push_back("upper=null");
      }

      if (expectedLowerBound.has_value()) {
        const auto& bound = expectedLowerBound.value();
        std::vector<std::string> values;
        for (size_t i = 0; i < bound->childrenSize(); ++i) {
          values.push_back(vectorToString(bound->childAt(i)));
        }
        parts.push_back(
            fmt::format("expected_lower=[{}]", folly::join(", ", values)));
      } else {
        parts.push_back("expected_lower=null");
      }

      if (expectedUpperBound.has_value()) {
        const auto& bound = expectedUpperBound.value();
        std::vector<std::string> values;
        for (size_t i = 0; i < bound->childrenSize(); ++i) {
          values.push_back(vectorToString(bound->childAt(i)));
        }
        parts.push_back(
            fmt::format("expected_upper=[{}]", folly::join(", ", values)));
      } else {
        parts.push_back("expected_upper=null");
      }

      return folly::join(", ", parts);
    }
  };

  // Test method to validate KeyEncoder encodeIndexBounds against expected
  // encoded bounds
  void testIndexBounds(const EncodeIndexBoundsTestCase& testCase) {
    IndexBounds indexBounds;
    indexBounds.indexColumns = testCase.indexColumns;
    indexBounds.lowerBound = testCase.lowerBound;
    indexBounds.upperBound = testCase.upperBound;

    ASSERT_TRUE(indexBounds.validate());

    const auto inputType = asRowType(indexBounds.type());
    const std::vector<velox::core::SortOrder> sortOrders(
        testCase.indexColumns.size(), testCase.sortOrder);
    auto keyEncoder = KeyEncoder::create(
        indexBounds.indexColumns, inputType, sortOrders, pool_.get());

    if (testCase.expectedFailure) {
      // For lower bound bump failures, expect exception to be thrown.
      VELOX_ASSERT_THROW(
          keyEncoder->encodeIndexBounds(indexBounds),
          "Failed to bump up lower bound");
      return;
    }

    const auto encodedBounds = keyEncoder->encodeIndexBounds(indexBounds);

    ASSERT_EQ(encodedBounds.size(), 1);

    const auto& result = encodedBounds[0];

    // Verify presence of keys matches expectations
    EXPECT_EQ(
        result.lowerKey.has_value(), testCase.expectedLowerBound.has_value());
    EXPECT_EQ(
        result.upperKey.has_value(), testCase.expectedUpperBound.has_value());

    // Encode expected bounds and verify they match
    if (testCase.expectedLowerBound.has_value()) {
      ASSERT_TRUE(result.lowerKey.has_value());
      std::vector<char> expectedBuffer;
      std::vector<std::string_view> expectedKeys;
      keyEncoder->encode(
          testCase.expectedLowerBound.value(),
          expectedKeys,
          [&expectedBuffer](size_t size) -> void* {
            expectedBuffer.resize(size);
            return expectedBuffer.data();
          });
      EXPECT_EQ(expectedKeys.size(), 1);
      EXPECT_EQ(result.lowerKey.value(), std::string(expectedKeys[0]));
    }

    if (testCase.expectedUpperBound.has_value()) {
      ASSERT_TRUE(result.upperKey.has_value());
      std::vector<char> expectedBuffer;
      std::vector<std::string_view> expectedKeys;
      keyEncoder->encode(
          testCase.expectedUpperBound.value(),
          expectedKeys,
          [&expectedBuffer](size_t size) -> void* {
            expectedBuffer.resize(size);
            return expectedBuffer.data();
          });
      EXPECT_EQ(expectedKeys.size(), 1);
      EXPECT_EQ(result.upperKey.value(), std::string(expectedKeys[0]));
    }
  }

  // Helper method to create test cases for all sort orders with sort
  // order-specific expected values
  std::vector<EncodeIndexBoundsTestCase> createIndexBoundEncodeTestCases(
      std::vector<std::string> indexColumns,
      std::optional<IndexBound> lowerBound,
      std::optional<IndexBound> upperBound,
      std::optional<velox::RowVectorPtr> ascNullsFirstExpectedLowerBound,
      std::optional<velox::RowVectorPtr> ascNullsFirstExpectedUpperBound,
      std::optional<velox::RowVectorPtr> ascNullsLastExpectedLowerBound,
      std::optional<velox::RowVectorPtr> ascNullsLastExpectedUpperBound,
      std::optional<velox::RowVectorPtr> descNullsFirstExpectedLowerBound,
      std::optional<velox::RowVectorPtr> descNullsFirstExpectedUpperBound,
      std::optional<velox::RowVectorPtr> descNullsLastExpectedLowerBound,
      std::optional<velox::RowVectorPtr> descNullsLastExpectedUpperBound) {
    const std::vector<velox::core::SortOrder> sortOrders = {
        velox::core::kAscNullsFirst,
        velox::core::kAscNullsLast,
        velox::core::kDescNullsFirst,
        velox::core::kDescNullsLast,
    };

    std::vector<EncodeIndexBoundsTestCase> testCases;
    testCases.reserve(sortOrders.size());

    for (const auto& sortOrder : sortOrders) {
      std::optional<velox::RowVectorPtr> expectedLowerBound;
      std::optional<velox::RowVectorPtr> expectedUpperBound;

      if (sortOrder.isAscending() && sortOrder.isNullsFirst()) {
        expectedLowerBound = ascNullsFirstExpectedLowerBound;
        expectedUpperBound = ascNullsFirstExpectedUpperBound;
      } else if (sortOrder.isAscending() && !sortOrder.isNullsFirst()) {
        expectedLowerBound = ascNullsLastExpectedLowerBound;
        expectedUpperBound = ascNullsLastExpectedUpperBound;
      } else if (!sortOrder.isAscending() && sortOrder.isNullsFirst()) {
        expectedLowerBound = descNullsFirstExpectedLowerBound;
        expectedUpperBound = descNullsFirstExpectedUpperBound;
      } else { // DESC NULLS_LAST
        expectedLowerBound = descNullsLastExpectedLowerBound;
        expectedUpperBound = descNullsLastExpectedUpperBound;
      }

      testCases.push_back({
          .indexColumns = indexColumns,
          .lowerBound = lowerBound,
          .upperBound = upperBound,
          .expectedLowerBound = expectedLowerBound,
          .expectedUpperBound = expectedUpperBound,
          .sortOrder = sortOrder,
      });
    }

    return testCases;
  }
};
} // namespace

TEST_F(KeyEncoderTest, indexBounds) {
  const auto boundRow = makeRowVector({makeNullableFlatVector<int32_t>({2})});
  // Test valid bounds with lower bound.
  IndexBounds boundsWithLower;
  boundsWithLower.indexColumns = {"c0"};
  boundsWithLower.lowerBound = IndexBound{boundRow, true};
  EXPECT_TRUE(boundsWithLower.validate());
  EXPECT_EQ(
      boundsWithLower.toString(),
      "IndexBounds{indexColumns=[c0], lowerBound=[0: {2}], upperBound=unbounded}");
  EXPECT_EQ(boundsWithLower.type()->toString(), boundRow->type()->toString());
  EXPECT_EQ(boundsWithLower.numRows(), 1);

  // Test valid bounds with upper bound.
  IndexBounds boundsWithUpper;
  boundsWithUpper.indexColumns = {"c0"};
  boundsWithUpper.upperBound = IndexBound{boundRow, false};
  EXPECT_TRUE(boundsWithUpper.validate());
  EXPECT_EQ(
      boundsWithUpper.toString(),
      "IndexBounds{indexColumns=[c0], lowerBound=unbounded, upperBound=(0: {2})}");
  EXPECT_EQ(boundsWithUpper.numRows(), 1);

  // Test valid bounds with both.
  IndexBounds boundsWithBoth;
  boundsWithBoth.indexColumns = {"c0"};
  boundsWithBoth.lowerBound = IndexBound{boundRow, true};
  boundsWithBoth.upperBound = IndexBound{boundRow, false};
  EXPECT_TRUE(boundsWithBoth.validate());
  EXPECT_EQ(
      boundsWithBoth.toString(),
      "IndexBounds{indexColumns=[c0], lowerBound=[0: {2}], upperBound=(0: {2})}");

  // Test invalid bounds with no bounds.
  IndexBounds noBounds;
  noBounds.indexColumns = {"c0"};
  EXPECT_FALSE(noBounds.validate());

  // Test multi-row bounds (valid with multi-row support).
  auto twoRowBound =
      makeRowVector({makeNullableFlatVector<int32_t>({2, std::nullopt})});
  IndexBounds twoRowBounds;
  twoRowBounds.indexColumns = {"c0"};
  twoRowBounds.lowerBound = IndexBound{twoRowBound, true};
  EXPECT_TRUE(twoRowBounds.validate());
  EXPECT_EQ(twoRowBounds.numRows(), 2);

  // Test invalid bounds with extra index columns.
  IndexBounds extraColBounds;
  extraColBounds.indexColumns = {"c0", "c1", "c2"};
  extraColBounds.lowerBound = IndexBound{boundRow, true};
  EXPECT_FALSE(extraColBounds.validate());

  extraColBounds.lowerBound = std::nullopt;
  extraColBounds.upperBound = IndexBound{boundRow, true};
  EXPECT_FALSE(extraColBounds.validate());

  const auto multiColumnBoundRow = makeRowVector(
      {makeNullableFlatVector<int32_t>({2}),
       makeNullableFlatVector<int32_t>({1})});
  IndexBounds multiColumnBounds;
  multiColumnBounds.indexColumns = {"c0", "c1"};
  multiColumnBounds.lowerBound = IndexBound{multiColumnBoundRow, true};
  multiColumnBounds.upperBound = IndexBound{multiColumnBoundRow, false};
  EXPECT_TRUE(multiColumnBounds.validate());
  EXPECT_EQ(
      multiColumnBounds.toString(),
      "IndexBounds{indexColumns=[c0, c1], lowerBound=[0: {2, 1}], upperBound=(0: {2, 1})}");

  IndexBounds missingColBounds;
  missingColBounds.indexColumns = {"c0"};
  missingColBounds.upperBound = IndexBound{multiColumnBoundRow, false};
  EXPECT_FALSE(missingColBounds.validate());

  // Test set() and clear() methods.
  const auto lowerRow = makeRowVector({makeNullableFlatVector<int32_t>({10})});
  const auto upperRow = makeRowVector({makeNullableFlatVector<int32_t>({20})});

  IndexBounds bounds;
  bounds.indexColumns = {"c0"};
  EXPECT_FALSE(bounds.validate());

  bounds.set(IndexBound{lowerRow, true}, IndexBound{upperRow, false});
  EXPECT_TRUE(bounds.validate());
  EXPECT_TRUE(bounds.lowerBound.has_value());
  EXPECT_TRUE(bounds.upperBound.has_value());
  EXPECT_TRUE(bounds.lowerBound->inclusive);
  EXPECT_FALSE(bounds.upperBound->inclusive);
  EXPECT_EQ(bounds.numRows(), 1);
  EXPECT_EQ(
      bounds.toString(),
      "IndexBounds{indexColumns=[c0], lowerBound=[0: {10}], upperBound=(0: {20})}");

  bounds.clear();
  EXPECT_FALSE(bounds.lowerBound.has_value());
  EXPECT_FALSE(bounds.upperBound.has_value());
  EXPECT_FALSE(bounds.validate());

  bounds.set(IndexBound{upperRow, false}, IndexBound{lowerRow, true});
  EXPECT_TRUE(bounds.validate());
  EXPECT_EQ(bounds.numRows(), 1);

  // Test numRows() with multiple row bounds.
  const auto multipleRows =
      makeRowVector({makeNullableFlatVector<int32_t>({1, 2, 3})});
  IndexBounds multiRowBounds;
  multiRowBounds.indexColumns = {"c0"};
  multiRowBounds.lowerBound = IndexBound{multipleRows, true};
  EXPECT_EQ(multiRowBounds.numRows(), 3);

  IndexBounds multiRowBothBounds;
  multiRowBothBounds.indexColumns = {"c0"};
  multiRowBothBounds.lowerBound = IndexBound{multipleRows, true};
  multiRowBothBounds.upperBound = IndexBound{multipleRows, false};
  EXPECT_EQ(multiRowBothBounds.numRows(), 3);
}

TEST_F(KeyEncoderTest, indexBoundsType) {
  struct TestCase {
    std::string name;
    std::vector<std::string> indexColumns;
    std::optional<RowVectorPtr> lowerBoundRow;
    std::optional<RowVectorPtr> upperBoundRow;
    bool lowerInclusive;
    bool upperInclusive;
    // Expected type is always the type from lowerBoundRow if present,
    // otherwise from upperBoundRow
  };

  std::vector<TestCase> testCases = {
      // Lower bound only - single column
      {
          .name = "lower_bound_only_single_column",
          .indexColumns = {"key"},
          .lowerBoundRow =
              makeRowVector({"key"}, {makeNullableFlatVector<int32_t>({42})}),
          .upperBoundRow = std::nullopt,
          .lowerInclusive = true,
          .upperInclusive = false,
      },
      // Upper bound only - single column
      {
          .name = "upper_bound_only_single_column",
          .indexColumns = {"key"},
          .lowerBoundRow = std::nullopt,
          .upperBoundRow =
              makeRowVector({"key"}, {makeNullableFlatVector<int64_t>({100})}),
          .lowerInclusive = false,
          .upperInclusive = true,
      },
      // Both bounds - single column
      {
          .name = "both_bounds_single_column",
          .indexColumns = {"col"},
          .lowerBoundRow =
              makeRowVector({"col"}, {makeNullableFlatVector<double>({1.5})}),
          .upperBoundRow =
              makeRowVector({"col"}, {makeNullableFlatVector<double>({10.5})}),
          .lowerInclusive = true,
          .upperInclusive = false,
      },
      // Lower bound only - multi column
      {
          .name = "lower_bound_only_multi_column",
          .indexColumns = {"col1", "col2"},
          .lowerBoundRow = makeRowVector(
              {"col1", "col2"},
              {makeNullableFlatVector<int64_t>({100}),
               makeNullableFlatVector<std::string>({"test"})}),
          .upperBoundRow = std::nullopt,
          .lowerInclusive = true,
          .upperInclusive = false,
      },
      // Upper bound only - multi column
      {
          .name = "upper_bound_only_multi_column",
          .indexColumns = {"a", "b", "c"},
          .lowerBoundRow = std::nullopt,
          .upperBoundRow = makeRowVector(
              {"a", "b", "c"},
              {makeNullableFlatVector<int32_t>({1}),
               makeNullableFlatVector<int32_t>({2}),
               makeNullableFlatVector<int32_t>({3})}),
          .lowerInclusive = false,
          .upperInclusive = true,
      },
      // Both bounds - multi column
      {
          .name = "both_bounds_multi_column",
          .indexColumns = {"x", "y"},
          .lowerBoundRow = makeRowVector(
              {"x", "y"},
              {makeNullableFlatVector<bool>({true}),
               makeNullableFlatVector<float>({1.0f})}),
          .upperBoundRow = makeRowVector(
              {"x", "y"},
              {makeNullableFlatVector<bool>({false}),
               makeNullableFlatVector<float>({9.0f})}),
          .lowerInclusive = false,
          .upperInclusive = false,
      },
      // Timestamp type
      {
          .name = "timestamp_type",
          .indexColumns = {"ts"},
          .lowerBoundRow = makeRowVector(
              {"ts"},
              {makeNullableFlatVector<Timestamp>({Timestamp(1000, 500)})}),
          .upperBoundRow = std::nullopt,
          .lowerInclusive = true,
          .upperInclusive = false,
      },
      // VARCHAR type
      {
          .name = "varchar_type",
          .indexColumns = {"name"},
          .lowerBoundRow = std::nullopt,
          .upperBoundRow = makeRowVector(
              {"name"}, {makeNullableFlatVector<std::string>({"hello"})}),
          .lowerInclusive = false,
          .upperInclusive = true,
      },
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.name);

    IndexBounds bounds;
    bounds.indexColumns = testCase.indexColumns;
    if (testCase.lowerBoundRow.has_value()) {
      bounds.lowerBound =
          IndexBound{testCase.lowerBoundRow.value(), testCase.lowerInclusive};
    }
    if (testCase.upperBoundRow.has_value()) {
      bounds.upperBound =
          IndexBound{testCase.upperBoundRow.value(), testCase.upperInclusive};
    }

    ASSERT_TRUE(bounds.validate());

    const auto type = bounds.type();
    ASSERT_NE(type, nullptr);
    EXPECT_EQ(type->kind(), TypeKind::ROW);

    // type() should return lowerBound's type when present, otherwise
    // upperBound's type
    const auto& expectedRow = testCase.lowerBoundRow.has_value()
        ? testCase.lowerBoundRow.value()
        : testCase.upperBoundRow.value();
    EXPECT_TRUE(type->equivalent(*expectedRow->type()));

    const auto rowType = asRowType(type);
    EXPECT_EQ(rowType->size(), testCase.indexColumns.size());
    for (size_t i = 0; i < testCase.indexColumns.size(); ++i) {
      EXPECT_EQ(rowType->nameOf(i), testCase.indexColumns[i]);
      EXPECT_EQ(
          rowType->childAt(i)->kind(), expectedRow->childAt(i)->type()->kind());
    }
  }
}

TEST_F(KeyEncoderTest, longTypeWithoutNulls) {
  struct {
    std::vector<std::vector<int64_t>> columnValues;

    std::string debugString() const {
      const size_t numRows = columnValues[0].size();
      std::vector<std::string> rows;
      rows.reserve(numRows);

      for (size_t row = 0; row < numRows; ++row) {
        std::vector<int64_t> rowValues;
        rowValues.reserve(columnValues.size());
        for (const auto& column : columnValues) {
          rowValues.push_back(column[row]);
        }
        rows.push_back(fmt::format("{{{}}}", fmt::join(rowValues, ", ")));
      }

      return fmt::format("[{}]", fmt::join(rows, ", "));
    }
  } testCases[] = {
      // Single row
      {{{1}, {2}, {3}}},
      // All same values
      {{{5, 5, 5}, {10, 10, 10}, {15, 15, 15}}},
      // Ascending order in first column
      {{{1, 2, 3, 4}, {100, 100, 100, 100}, {50, 50, 50, 50}}},
      // Descending order in first column
      {{{10, 9, 8, 7}, {0, 0, 0, 0}, {5, 5, 5, 5}}},
      // Mixed positive and negative values
      {{{-5, 0, 5}, {-10, -5, 0}, {10, 5, 0}}},
      // Edge values: min and max int64_t
      {{{std::numeric_limits<int64_t>::min(),
         std::numeric_limits<int64_t>::max(),
         0},
        {-1, 0, 1},
        {100, 200, 300}}},
      // All zeros
      {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
      // Duplicate values across rows
      {{{1, 2, 1, 2}, {3, 3, 4, 4}, {5, 5, 5, 5}}},
      // Large spread of values
      {{{-1000000, -1, 0, 1, 1000000},
        {999999, 500000, 0, -500000, -999999},
        {42, 42, 42, 42, 42}}},
      // Values that differ only in last column
      {{{1, 1, 1}, {2, 2, 2}, {3, 4, 5}}},
      // Values that differ only in first column
      {{{1, 2, 3}, {100, 100, 100}, {200, 200, 200}}},
      // Alternating pattern
      {{{1, -1, 1, -1, 1, -1}, {2, -2, 2, -2, 2, -2}, {3, -3, 3, -3, 3, -3}}},
      // Sequential values
      {{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
        {20, 21, 22, 23, 24, 25, 26, 27, 28, 29}}},
      // Powers of 2
      {{{1, 2, 4, 8, 16, 32, 64},
        {128, 256, 512, 1024, 2048, 4096, 8192},
        {16384, 32768, 65536, 131072, 262144, 524288, 1048576}}},
      // Negative progression
      {{{-1, -2, -4, -8}, {-16, -32, -64, -128}, {0, 0, 0, 0}}},
      // Mix of edge cases and regular values
      {{{std::numeric_limits<int64_t>::min(), -1000, -1, 0},
        {1,
         1000,
         std::numeric_limits<int64_t>::max(),
         std::numeric_limits<int64_t>::max()},
        {42, 43, 44, 45}}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ASSERT_EQ(testCase.columnValues.size(), 3);
    const auto input = makeRowVector(
        {makeFlatVector<int64_t>(testCase.columnValues[0]),
         makeFlatVector<int64_t>(testCase.columnValues[1]),
         makeFlatVector<int64_t>(testCase.columnValues[2])});
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};

    // Test all four sort orders
    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, longTypeWithNulls) {
  struct {
    std::vector<std::vector<std::optional<int64_t>>> columnValues;

    std::string debugString() const {
      const size_t numRows = columnValues[0].size();
      std::vector<std::string> rows;
      rows.reserve(numRows);

      for (size_t row = 0; row < numRows; ++row) {
        std::vector<std::string> rowValues;
        rowValues.reserve(columnValues.size());
        for (const auto& column : columnValues) {
          if (column[row].has_value()) {
            rowValues.push_back(std::to_string(column[row].value()));
          } else {
            rowValues.push_back("null");
          }
        }
        rows.push_back(fmt::format("{{{}}}", fmt::join(rowValues, ", ")));
      }

      return fmt::format("[{}]", fmt::join(rows, ", "));
    }
  } testCases[] = {
      // Single row with null
      {{{std::nullopt}, {1}, {2}}},
      // Single row without null (nullable type)
      {{{1}, {2}, {3}}},
      // All nulls
      {{{std::nullopt, std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt, std::nullopt}}},
      // Null at beginning of first column
      {{{std::nullopt, 1, 2}, {10, 20, 30}, {100, 200, 300}}},
      // Null at end of first column
      {{{1, 2, std::nullopt}, {10, 20, 30}, {100, 200, 300}}},
      // Null in middle of first column
      {{{1, std::nullopt, 2}, {10, 20, 30}, {100, 200, 300}}},
      // Nulls in all positions across different columns
      {{{std::nullopt, 1, 2, 3},
        {1, std::nullopt, 2, 3},
        {1, 2, std::nullopt, 3}}},
      // Multiple nulls in same row
      {{{std::nullopt, 1, 2}, {std::nullopt, 1, 2}, {std::nullopt, 1, 2}}},
      // Alternating null and non-null in first column
      {{{std::nullopt, 1, std::nullopt, 2, std::nullopt, 3},
        {10, 10, 10, 10, 10, 10},
        {20, 20, 20, 20, 20, 20}}},
      // Same values, different null positions
      {{{1, 1, std::nullopt}, {2, std::nullopt, 2}, {3, 3, 3}}},
      // Nulls with edge values
      {{{std::nullopt,
         std::numeric_limits<int64_t>::min(),
         std::numeric_limits<int64_t>::max()},
        {0, std::nullopt, 0},
        {100, 200, std::nullopt}}},
      // Nulls with zeros
      {{{std::nullopt, 0, std::nullopt}, {0, std::nullopt, 0}, {0, 0, 0}}},
      // Nulls with positive and negative values
      {{{std::nullopt, -5, 5},
        {-10, std::nullopt, 10},
        {std::nullopt, -15, std::nullopt}}},
      // Duplicate non-null values with nulls
      {{{1, 1, std::nullopt, std::nullopt},
        {2, 2, 2, std::nullopt},
        {3, std::nullopt, 3, 3}}},
      // Ascending order with nulls scattered
      {{{std::nullopt, 1, 2, 3, std::nullopt},
        {10, std::nullopt, 20, std::nullopt, 30},
        {std::nullopt, std::nullopt, 100, 200, 300}}},
      // Descending order with nulls scattered
      {{{std::nullopt, 3, 2, 1, std::nullopt},
        {30, std::nullopt, 20, std::nullopt, 10},
        {std::nullopt, std::nullopt, 300, 200, 100}}},
      // Only first column has nulls
      {{{std::nullopt, std::nullopt, 1, 2}, {10, 20, 30, 40}, {5, 6, 7, 8}}},
      // Only middle column has nulls
      {{{1, 2, 3, 4},
        {std::nullopt, std::nullopt, 10, 20},
        {100, 200, 300, 400}}},
      // Only last column has nulls
      {{{1, 2, 3, 4},
        {10, 20, 30, 40},
        {std::nullopt, std::nullopt, 100, 200}}},
      // All columns have at least one null
      {{{std::nullopt, 1, 2, 3, 4},
        {1, std::nullopt, 2, 3, 4},
        {1, 2, std::nullopt, 3, 4}}},
      // Null vs non-null with same other column values
      {{{std::nullopt, 1}, {100, 100}, {200, 200}}},
      // Large spread with nulls
      {{{std::nullopt, -1000000, 0, 1000000, std::nullopt},
        {999999, std::nullopt, 0, std::nullopt, -999999},
        {std::nullopt, 42, std::nullopt, 42, std::nullopt}}},
      // Sequential with nulls at boundaries
      {{{std::nullopt, 0, 1, 2, 3, std::nullopt},
        {std::nullopt, 10, 11, 12, 13, std::nullopt},
        {std::nullopt, 20, 21, 22, 23, std::nullopt}}},
      // Powers of 2 with nulls
      {{{std::nullopt, 1, 2, 4, 8, std::nullopt},
        {16, std::nullopt, 32, 64, std::nullopt, 128},
        {std::nullopt, 256, std::nullopt, 512, std::nullopt, 1024}}},
      // Mix of nulls and regular values testing sort order edge cases
      {{{std::nullopt, std::nullopt, 0, 0},
        {std::nullopt, 1, std::nullopt, 1},
        {2, std::nullopt, 2, std::nullopt}}},
      // All max values
      {{{std::numeric_limits<int64_t>::max(),
         std::numeric_limits<int64_t>::max(),
         std::numeric_limits<int64_t>::max()},
        {std::numeric_limits<int64_t>::max(),
         std::numeric_limits<int64_t>::max(),
         std::numeric_limits<int64_t>::max()},
        {std::numeric_limits<int64_t>::max(),
         std::numeric_limits<int64_t>::max(),
         std::numeric_limits<int64_t>::max()}}},
      // All min values
      {{{std::numeric_limits<int64_t>::min(),
         std::numeric_limits<int64_t>::min(),
         std::numeric_limits<int64_t>::min()},
        {std::numeric_limits<int64_t>::min(),
         std::numeric_limits<int64_t>::min(),
         std::numeric_limits<int64_t>::min()},
        {std::numeric_limits<int64_t>::min(),
         std::numeric_limits<int64_t>::min(),
         std::numeric_limits<int64_t>::min()}}},
      // Max values with nulls
      {{{std::nullopt, std::numeric_limits<int64_t>::max(), std::nullopt},
        {std::numeric_limits<int64_t>::max(),
         std::nullopt,
         std::numeric_limits<int64_t>::max()},
        {std::nullopt, std::nullopt, std::numeric_limits<int64_t>::max()}}},
      // Min values with nulls
      {{{std::nullopt, std::numeric_limits<int64_t>::min(), std::nullopt},
        {std::numeric_limits<int64_t>::min(),
         std::nullopt,
         std::numeric_limits<int64_t>::min()},
        {std::nullopt, std::nullopt, std::numeric_limits<int64_t>::min()}}},
      // Mix of max and min values
      {{{std::numeric_limits<int64_t>::max(),
         std::numeric_limits<int64_t>::min(),
         std::numeric_limits<int64_t>::max()},
        {std::numeric_limits<int64_t>::min(),
         std::numeric_limits<int64_t>::max(),
         std::numeric_limits<int64_t>::min()},
        {std::numeric_limits<int64_t>::max(),
         std::numeric_limits<int64_t>::min(),
         std::numeric_limits<int64_t>::max()}}},
      // Mix of max, min, and nulls
      {{{std::nullopt,
         std::numeric_limits<int64_t>::max(),
         std::numeric_limits<int64_t>::min()},
        {std::numeric_limits<int64_t>::max(),
         std::nullopt,
         std::numeric_limits<int64_t>::min()},
        {std::numeric_limits<int64_t>::min(),
         std::numeric_limits<int64_t>::max(),
         std::nullopt}}},
      // Max in first column only
      {{{std::numeric_limits<int64_t>::max(),
         std::numeric_limits<int64_t>::max(),
         1,
         2},
        {100, 200, 300, 400},
        {5, 6, 7, 8}}},
      // Min in first column only
      {{{std::numeric_limits<int64_t>::min(),
         std::numeric_limits<int64_t>::min(),
         1,
         2},
        {100, 200, 300, 400},
        {5, 6, 7, 8}}},
      // Max and min in different columns
      {{{std::numeric_limits<int64_t>::max(), 1, 2},
        {1, std::numeric_limits<int64_t>::min(), 2},
        {1, 2, std::numeric_limits<int64_t>::max()}}},
      // Adjacent to max (max-1, max)
      {{{std::numeric_limits<int64_t>::max() - 1,
         std::numeric_limits<int64_t>::max(),
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int64_t>::max() - 1,
         std::numeric_limits<int64_t>::max()},
        {std::numeric_limits<int64_t>::max(),
         std::nullopt,
         std::numeric_limits<int64_t>::max() - 1}}},
      // Adjacent to min (min, min+1)
      {{{std::numeric_limits<int64_t>::min(),
         std::numeric_limits<int64_t>::min() + 1,
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int64_t>::min(),
         std::numeric_limits<int64_t>::min() + 1},
        {std::numeric_limits<int64_t>::min() + 1,
         std::nullopt,
         std::numeric_limits<int64_t>::min()}}},
      // Boundary transitions with nulls (min, null, 0, null, max)
      {{{std::numeric_limits<int64_t>::min(),
         std::nullopt,
         0,
         std::nullopt,
         std::numeric_limits<int64_t>::max()},
        {std::nullopt,
         std::numeric_limits<int64_t>::min(),
         std::nullopt,
         std::numeric_limits<int64_t>::max(),
         std::nullopt},
        {0, 0, 0, 0, 0}}},
      // Duplicate max values with nulls
      {{{std::numeric_limits<int64_t>::max(),
         std::numeric_limits<int64_t>::max(),
         std::nullopt,
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int64_t>::max(),
         std::numeric_limits<int64_t>::max(),
         std::nullopt},
        {std::numeric_limits<int64_t>::max(),
         std::nullopt,
         std::nullopt,
         std::numeric_limits<int64_t>::max()}}},
      // Duplicate min values with nulls
      {{{std::numeric_limits<int64_t>::min(),
         std::numeric_limits<int64_t>::min(),
         std::nullopt,
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int64_t>::min(),
         std::numeric_limits<int64_t>::min(),
         std::nullopt},
        {std::numeric_limits<int64_t>::min(),
         std::nullopt,
         std::nullopt,
         std::numeric_limits<int64_t>::min()}}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ASSERT_EQ(testCase.columnValues.size(), 3);
    const auto input = makeRowVector(
        {makeNullableFlatVector<int64_t>(testCase.columnValues[0]),
         makeNullableFlatVector<int64_t>(testCase.columnValues[1]),
         makeNullableFlatVector<int64_t>(testCase.columnValues[2])});
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};

    // Test all four sort orders (ASC/DESC x NULLS FIRST/NULLS LAST)
    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, integerTypeWithoutNulls) {
  struct {
    std::vector<std::vector<int32_t>> columnValues;

    std::string debugString() const {
      const size_t numRows = columnValues[0].size();
      std::vector<std::string> rows;
      rows.reserve(numRows);

      for (size_t row = 0; row < numRows; ++row) {
        std::vector<int32_t> rowValues;
        rowValues.reserve(columnValues.size());
        for (const auto& column : columnValues) {
          rowValues.push_back(column[row]);
        }
        rows.push_back(fmt::format("{{{}}}", fmt::join(rowValues, ", ")));
      }

      return fmt::format("[{}]", fmt::join(rows, ", "));
    }
  } testCases[] = {
      // Single row
      {{{1}, {2}, {3}}},
      // All same values
      {{{5, 5, 5}, {10, 10, 10}, {15, 15, 15}}},
      // Ascending order
      {{{1, 2, 3, 4}, {100, 100, 100, 100}, {50, 50, 50, 50}}},
      // Descending order
      {{{10, 9, 8, 7}, {0, 0, 0, 0}, {5, 5, 5, 5}}},
      // Mixed positive and negative values
      {{{-5, 0, 5}, {-10, -5, 0}, {10, 5, 0}}},
      // Edge values: min and max int32_t
      {{{std::numeric_limits<int32_t>::min(),
         std::numeric_limits<int32_t>::max(),
         0},
        {-1, 0, 1},
        {100, 200, 300}}},
      // All zeros
      {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
      // Duplicate values
      {{{1, 2, 1, 2}, {3, 3, 4, 4}, {5, 5, 5, 5}}},
      // Large spread
      {{{-1000000, -1, 0, 1, 1000000},
        {999999, 500000, 0, -500000, -999999},
        {42, 42, 42, 42, 42}}},
      // Values differing only in last column
      {{{1, 1, 1}, {2, 2, 2}, {3, 4, 5}}},
      // Values differing only in first column
      {{{1, 2, 3}, {100, 100, 100}, {200, 200, 200}}},
      // Alternating pattern
      {{{1, -1, 1, -1}, {2, -2, 2, -2}, {3, -3, 3, -3}}},
      // All max values
      {{{std::numeric_limits<int32_t>::max(),
         std::numeric_limits<int32_t>::max()},
        {std::numeric_limits<int32_t>::max(),
         std::numeric_limits<int32_t>::max()},
        {std::numeric_limits<int32_t>::max(),
         std::numeric_limits<int32_t>::max()}}},
      // All min values
      {{{std::numeric_limits<int32_t>::min(),
         std::numeric_limits<int32_t>::min()},
        {std::numeric_limits<int32_t>::min(),
         std::numeric_limits<int32_t>::min()},
        {std::numeric_limits<int32_t>::min(),
         std::numeric_limits<int32_t>::min()}}},
      // Mix of max and min
      {{{std::numeric_limits<int32_t>::max(),
         std::numeric_limits<int32_t>::min()},
        {std::numeric_limits<int32_t>::min(),
         std::numeric_limits<int32_t>::max()},
        {0, 0}}},
      // Adjacent to max (max-1, max)
      {{{std::numeric_limits<int32_t>::max() - 1,
         std::numeric_limits<int32_t>::max()},
        {std::numeric_limits<int32_t>::max(),
         std::numeric_limits<int32_t>::max() - 1},
        {0, 1}}},
      // Adjacent to min (min, min+1)
      {{{std::numeric_limits<int32_t>::min(),
         std::numeric_limits<int32_t>::min() + 1},
        {std::numeric_limits<int32_t>::min() + 1,
         std::numeric_limits<int32_t>::min()},
        {0, 1}}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ASSERT_EQ(testCase.columnValues.size(), 3);
    const auto input = makeRowVector(
        {makeFlatVector<int32_t>(testCase.columnValues[0]),
         makeFlatVector<int32_t>(testCase.columnValues[1]),
         makeFlatVector<int32_t>(testCase.columnValues[2])});
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};

    // Test all four sort orders
    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, integerTypeWithNulls) {
  struct {
    std::vector<std::vector<std::optional<int32_t>>> columnValues;

    std::string debugString() const {
      const size_t numRows = columnValues[0].size();
      std::vector<std::string> rows;
      rows.reserve(numRows);

      for (size_t row = 0; row < numRows; ++row) {
        std::vector<std::string> rowValues;
        rowValues.reserve(columnValues.size());
        for (const auto& column : columnValues) {
          if (column[row].has_value()) {
            rowValues.push_back(std::to_string(column[row].value()));
          } else {
            rowValues.push_back("null");
          }
        }
        rows.push_back(fmt::format("{{{}}}", fmt::join(rowValues, ", ")));
      }

      return fmt::format("[{}]", fmt::join(rows, ", "));
    }
  } testCases[] = {
      // Single row with null
      {{{std::nullopt}, {1}, {2}}},
      // Single row without null (nullable type)
      {{{1}, {2}, {3}}},
      // All nulls
      {{{std::nullopt, std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt, std::nullopt}}},
      // Null at beginning with max value
      {{{std::nullopt, 1, std::numeric_limits<int32_t>::max()},
        {10, 20, 30},
        {100, 200, 300}}},
      // Null at end with min value
      {{{1, std::numeric_limits<int32_t>::min(), std::nullopt},
        {10, 20, 30},
        {100, 200, 300}}},
      // Null in middle
      {{{1, std::nullopt, 2}, {10, 20, 30}, {100, 200, 300}}},
      // Nulls in all positions across different columns with edge values
      {{{std::nullopt, 1, std::numeric_limits<int32_t>::max(), 3},
        {1, std::nullopt, 2, std::numeric_limits<int32_t>::min()},
        {std::numeric_limits<int32_t>::min(), 2, std::nullopt, 3}}},
      // Multiple nulls in same row
      {{{std::nullopt, 1, 2}, {std::nullopt, 1, 2}, {std::nullopt, 1, 2}}},
      // Alternating null and non-null with boundaries
      {{{std::nullopt,
         std::numeric_limits<int32_t>::max(),
         std::nullopt,
         2,
         std::nullopt,
         std::numeric_limits<int32_t>::min()},
        {10, 10, 10, 10, 10, 10},
        {20, 20, 20, 20, 20, 20}}},
      // Same values, different null positions
      {{{1, 1, std::nullopt}, {2, std::nullopt, 2}, {3, 3, 3}}},
      // Mixed edge values with nulls
      {{{std::nullopt,
         std::numeric_limits<int32_t>::min(),
         std::numeric_limits<int32_t>::max()},
        {0, std::nullopt, 0},
        {100, 200, std::nullopt}}},
      // Nulls with zeros and boundaries
      {{{std::nullopt, 0, std::numeric_limits<int32_t>::max()},
        {0, std::nullopt, std::numeric_limits<int32_t>::min()},
        {0, 0, 0}}},
      // Nulls with positive and negative including extremes
      {{{std::nullopt, -5, std::numeric_limits<int32_t>::max()},
        {std::numeric_limits<int32_t>::min(), std::nullopt, 10},
        {std::nullopt, -15, std::nullopt}}},
      // Duplicate non-null values with nulls
      {{{1, 1, std::nullopt, std::nullopt},
        {2, 2, 2, std::nullopt},
        {3, std::nullopt, 3, 3}}},
      // Ascending order with nulls and max values
      {{{std::nullopt, 1, 2, std::numeric_limits<int32_t>::max(), std::nullopt},
        {10, std::nullopt, 20, std::nullopt, 30},
        {std::nullopt, std::nullopt, 100, 200, 300}}},
      // Descending order with nulls and min values
      {{{std::nullopt,
         std::numeric_limits<int32_t>::max(),
         2,
         std::numeric_limits<int32_t>::min(),
         std::nullopt},
        {30, std::nullopt, 20, std::nullopt, 10},
        {std::nullopt, std::nullopt, 300, 200, 100}}},
      // Only first column has nulls with boundaries
      {{{std::nullopt,
         std::nullopt,
         std::numeric_limits<int32_t>::min(),
         std::numeric_limits<int32_t>::max()},
        {10, 20, 30, 40},
        {5, 6, 7, 8}}},
      // Only middle column has nulls
      {{{1, 2, 3, 4},
        {std::nullopt, std::nullopt, 10, 20},
        {100, 200, 300, 400}}},
      // Only last column has nulls with max
      {{{1, 2, 3, std::numeric_limits<int32_t>::max()},
        {10, 20, 30, 40},
        {std::nullopt,
         std::nullopt,
         100,
         std::numeric_limits<int32_t>::min()}}},
      // All columns have at least one null mixed with extremes
      {{{std::nullopt, std::numeric_limits<int32_t>::max(), 2, 3, 4},
        {1, std::nullopt, std::numeric_limits<int32_t>::min(), 3, 4},
        {1, 2, std::nullopt, std::numeric_limits<int32_t>::max(), 4}}},
      // Null vs non-null with same other column values
      {{{std::nullopt, 1}, {100, 100}, {200, 200}}},
      // Large spread with nulls and boundaries
      {{{std::nullopt,
         -1000000,
         std::numeric_limits<int32_t>::min(),
         std::numeric_limits<int32_t>::max(),
         std::nullopt},
        {999999, std::nullopt, 0, std::nullopt, -999999},
        {std::nullopt, 42, std::nullopt, 42, std::nullopt}}},
      // Sequential with nulls at boundaries
      {{{std::nullopt,
         0,
         1,
         std::numeric_limits<int32_t>::min(),
         std::numeric_limits<int32_t>::max(),
         std::nullopt},
        {std::nullopt, 10, 11, 12, 13, std::nullopt},
        {std::nullopt, 20, 21, 22, 23, std::nullopt}}},
      // Mix of nulls and regular values testing sort order
      {{{std::nullopt, std::nullopt, 0, 0},
        {std::nullopt, 1, std::nullopt, 1},
        {2, std::nullopt, 2, std::nullopt}}},
      // Max and min together in different positions
      {{{std::numeric_limits<int32_t>::max(),
         std::numeric_limits<int32_t>::min(),
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int32_t>::max(),
         std::numeric_limits<int32_t>::min()},
        {std::numeric_limits<int32_t>::min(),
         std::nullopt,
         std::numeric_limits<int32_t>::max()}}},
      // Adjacent to max with nulls (max-1, max)
      {{{std::numeric_limits<int32_t>::max() - 1,
         std::numeric_limits<int32_t>::max(),
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int32_t>::max() - 1,
         std::numeric_limits<int32_t>::max()},
        {std::numeric_limits<int32_t>::max(),
         std::nullopt,
         std::numeric_limits<int32_t>::max() - 1}}},
      // Adjacent to min with nulls (min, min+1)
      {{{std::numeric_limits<int32_t>::min(),
         std::numeric_limits<int32_t>::min() + 1,
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int32_t>::min(),
         std::numeric_limits<int32_t>::min() + 1},
        {std::numeric_limits<int32_t>::min() + 1,
         std::nullopt,
         std::numeric_limits<int32_t>::min()}}},
      // Boundary transitions with nulls (min, null, 0, null, max)
      {{{std::numeric_limits<int32_t>::min(),
         std::nullopt,
         0,
         std::nullopt,
         std::numeric_limits<int32_t>::max()},
        {std::nullopt,
         std::numeric_limits<int32_t>::min(),
         std::nullopt,
         std::numeric_limits<int32_t>::max(),
         std::nullopt},
        {0, 0, 0, 0, 0}}},
      // Duplicate max values with nulls
      {{{std::numeric_limits<int32_t>::max(),
         std::numeric_limits<int32_t>::max(),
         std::nullopt,
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int32_t>::max(),
         std::numeric_limits<int32_t>::max(),
         std::nullopt},
        {std::numeric_limits<int32_t>::max(),
         std::nullopt,
         std::nullopt,
         std::numeric_limits<int32_t>::max()}}},
      // Duplicate min values with nulls
      {{{std::numeric_limits<int32_t>::min(),
         std::numeric_limits<int32_t>::min(),
         std::nullopt,
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int32_t>::min(),
         std::numeric_limits<int32_t>::min(),
         std::nullopt},
        {std::numeric_limits<int32_t>::min(),
         std::nullopt,
         std::nullopt,
         std::numeric_limits<int32_t>::min()}}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ASSERT_EQ(testCase.columnValues.size(), 3);
    const auto input = makeRowVector(
        {makeNullableFlatVector<int32_t>(testCase.columnValues[0]),
         makeNullableFlatVector<int32_t>(testCase.columnValues[1]),
         makeNullableFlatVector<int32_t>(testCase.columnValues[2])});
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};

    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, shortTypeWithoutNulls) {
  struct {
    std::vector<std::vector<int16_t>> columnValues;

    std::string debugString() const {
      const size_t numRows = columnValues[0].size();
      std::vector<std::string> rows;
      rows.reserve(numRows);

      for (size_t row = 0; row < numRows; ++row) {
        std::vector<int16_t> rowValues;
        rowValues.reserve(columnValues.size());
        for (const auto& column : columnValues) {
          rowValues.push_back(column[row]);
        }
        rows.push_back(fmt::format("{{{}}}", fmt::join(rowValues, ", ")));
      }

      return fmt::format("[{}]", fmt::join(rows, ", "));
    }
  } testCases[] = {
      // Single row
      {{{1}, {2}, {3}}},
      // All same values
      {{{5, 5, 5}, {10, 10, 10}, {15, 15, 15}}},
      // Ascending order
      {{{1, 2, 3, 4}, {100, 100, 100, 100}, {50, 50, 50, 50}}},
      // Descending order
      {{{10, 9, 8, 7}, {0, 0, 0, 0}, {5, 5, 5, 5}}},
      // Mixed positive and negative values
      {{{-5, 0, 5}, {-10, -5, 0}, {10, 5, 0}}},
      // Edge values: min and max int16_t
      {{{std::numeric_limits<int16_t>::min(),
         std::numeric_limits<int16_t>::max(),
         0},
        {-1, 0, 1},
        {100, 200, 300}}},
      // All zeros
      {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
      // Duplicate values
      {{{1, 2, 1, 2}, {3, 3, 4, 4}, {5, 5, 5, 5}}},
      // Large spread (within int16 range)
      {{{-30000, -1, 0, 1, 30000},
        {20000, 10000, 0, -10000, -20000},
        {42, 42, 42, 42, 42}}},
      // Values differing only in last column
      {{{1, 1, 1}, {2, 2, 2}, {3, 4, 5}}},
      // Values differing only in first column
      {{{1, 2, 3}, {100, 100, 100}, {200, 200, 200}}},
      // Alternating pattern
      {{{1, -1, 1, -1}, {2, -2, 2, -2}, {3, -3, 3, -3}}},
      // All max values
      {{{std::numeric_limits<int16_t>::max(),
         std::numeric_limits<int16_t>::max()},
        {std::numeric_limits<int16_t>::max(),
         std::numeric_limits<int16_t>::max()},
        {std::numeric_limits<int16_t>::max(),
         std::numeric_limits<int16_t>::max()}}},
      // All min values
      {{{std::numeric_limits<int16_t>::min(),
         std::numeric_limits<int16_t>::min()},
        {std::numeric_limits<int16_t>::min(),
         std::numeric_limits<int16_t>::min()},
        {std::numeric_limits<int16_t>::min(),
         std::numeric_limits<int16_t>::min()}}},
      // Mix of max and min
      {{{std::numeric_limits<int16_t>::max(),
         std::numeric_limits<int16_t>::min()},
        {std::numeric_limits<int16_t>::min(),
         std::numeric_limits<int16_t>::max()},
        {0, 0}}},
      // Adjacent to max (max-1, max)
      {{{std::numeric_limits<int16_t>::max() - 1,
         std::numeric_limits<int16_t>::max()},
        {std::numeric_limits<int16_t>::max(),
         std::numeric_limits<int16_t>::max() - 1},
        {0, 1}}},
      // Adjacent to min (min, min+1)
      {{{std::numeric_limits<int16_t>::min(),
         std::numeric_limits<int16_t>::min() + 1},
        {std::numeric_limits<int16_t>::min() + 1,
         std::numeric_limits<int16_t>::min()},
        {0, 1}}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ASSERT_EQ(testCase.columnValues.size(), 3);
    const auto input = makeRowVector(
        {makeFlatVector<int16_t>(testCase.columnValues[0]),
         makeFlatVector<int16_t>(testCase.columnValues[1]),
         makeFlatVector<int16_t>(testCase.columnValues[2])});
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};

    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, shortTypeWithNulls) {
  struct {
    std::vector<std::vector<std::optional<int16_t>>> columnValues;

    std::string debugString() const {
      const size_t numRows = columnValues[0].size();
      std::vector<std::string> rows;
      rows.reserve(numRows);

      for (size_t row = 0; row < numRows; ++row) {
        std::vector<std::string> rowValues;
        rowValues.reserve(columnValues.size());
        for (const auto& column : columnValues) {
          if (column[row].has_value()) {
            rowValues.push_back(std::to_string(column[row].value()));
          } else {
            rowValues.push_back("null");
          }
        }
        rows.push_back(fmt::format("{{{}}}", fmt::join(rowValues, ", ")));
      }

      return fmt::format("[{}]", fmt::join(rows, ", "));
    }
  } testCases[] = {
      // Single row with null
      {{{std::nullopt}, {1}, {2}}},
      // Single row without null (nullable type)
      {{{1}, {2}, {3}}},
      // All nulls
      {{{std::nullopt, std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt, std::nullopt}}},
      // Null at beginning with max value
      {{{std::nullopt, 1, std::numeric_limits<int16_t>::max()},
        {10, 20, 30},
        {100, 200, 300}}},
      // Null at end with min value
      {{{1, std::numeric_limits<int16_t>::min(), std::nullopt},
        {10, 20, 30},
        {100, 200, 300}}},
      // Null in middle
      {{{1, std::nullopt, 2}, {10, 20, 30}, {100, 200, 300}}},
      // Nulls in all positions across different columns with edge values
      {{{std::nullopt, 1, std::numeric_limits<int16_t>::max(), 3},
        {1, std::nullopt, 2, std::numeric_limits<int16_t>::min()},
        {std::numeric_limits<int16_t>::min(), 2, std::nullopt, 3}}},
      // Multiple nulls in same row
      {{{std::nullopt, 1, 2}, {std::nullopt, 1, 2}, {std::nullopt, 1, 2}}},
      // Alternating null and non-null with boundaries
      {{{std::nullopt,
         std::numeric_limits<int16_t>::max(),
         std::nullopt,
         2,
         std::nullopt,
         std::numeric_limits<int16_t>::min()},
        {10, 10, 10, 10, 10, 10},
        {20, 20, 20, 20, 20, 20}}},
      // Same values, different null positions
      {{{1, 1, std::nullopt}, {2, std::nullopt, 2}, {3, 3, 3}}},
      // Mixed edge values with nulls
      {{{std::nullopt,
         std::numeric_limits<int16_t>::min(),
         std::numeric_limits<int16_t>::max()},
        {0, std::nullopt, 0},
        {100, 200, std::nullopt}}},
      // Nulls with zeros and boundaries
      {{{std::nullopt, 0, std::numeric_limits<int16_t>::max()},
        {0, std::nullopt, std::numeric_limits<int16_t>::min()},
        {0, 0, 0}}},
      // Nulls with positive and negative including extremes
      {{{std::nullopt, -5, std::numeric_limits<int16_t>::max()},
        {std::numeric_limits<int16_t>::min(), std::nullopt, 10},
        {std::nullopt, -15, std::nullopt}}},
      // Duplicate non-null values with nulls
      {{{1, 1, std::nullopt, std::nullopt},
        {2, 2, 2, std::nullopt},
        {3, std::nullopt, 3, 3}}},
      // Ascending order with nulls and max values
      {{{std::nullopt, 1, 2, std::numeric_limits<int16_t>::max(), std::nullopt},
        {10, std::nullopt, 20, std::nullopt, 30},
        {std::nullopt, std::nullopt, 100, 200, 300}}},
      // Descending order with nulls and min values
      {{{std::nullopt,
         std::numeric_limits<int16_t>::max(),
         2,
         std::numeric_limits<int16_t>::min(),
         std::nullopt},
        {30, std::nullopt, 20, std::nullopt, 10},
        {std::nullopt, std::nullopt, 300, 200, 100}}},
      // Only first column has nulls with boundaries
      {{{std::nullopt,
         std::nullopt,
         std::numeric_limits<int16_t>::min(),
         std::numeric_limits<int16_t>::max()},
        {10, 20, 30, 40},
        {5, 6, 7, 8}}},
      // Only middle column has nulls
      {{{1, 2, 3, 4},
        {std::nullopt, std::nullopt, 10, 20},
        {100, 200, 300, 400}}},
      // Only last column has nulls with max
      {{{1, 2, 3, std::numeric_limits<int16_t>::max()},
        {10, 20, 30, 40},
        {std::nullopt,
         std::nullopt,
         100,
         std::numeric_limits<int16_t>::min()}}},
      // All columns have at least one null mixed with extremes
      {{{std::nullopt, std::numeric_limits<int16_t>::max(), 2, 3, 4},
        {1, std::nullopt, std::numeric_limits<int16_t>::min(), 3, 4},
        {1, 2, std::nullopt, std::numeric_limits<int16_t>::max(), 4}}},
      // Null vs non-null with same other column values
      {{{std::nullopt, 1}, {100, 100}, {200, 200}}},
      // Large spread with nulls and boundaries (within int16 range)
      {{{std::nullopt,
         -30000,
         std::numeric_limits<int16_t>::min(),
         std::numeric_limits<int16_t>::max(),
         std::nullopt},
        {20000, std::nullopt, 0, std::nullopt, -20000},
        {std::nullopt, 42, std::nullopt, 42, std::nullopt}}},
      // Mix of nulls and regular values testing sort order
      {{{std::nullopt, std::nullopt, 0, 0},
        {std::nullopt, 1, std::nullopt, 1},
        {2, std::nullopt, 2, std::nullopt}}},
      // Max and min together in different positions
      {{{std::numeric_limits<int16_t>::max(),
         std::numeric_limits<int16_t>::min(),
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int16_t>::max(),
         std::numeric_limits<int16_t>::min()},
        {std::numeric_limits<int16_t>::min(),
         std::nullopt,
         std::numeric_limits<int16_t>::max()}}},
      // Adjacent to max with nulls (max-1, max)
      {{{std::numeric_limits<int16_t>::max() - 1,
         std::numeric_limits<int16_t>::max(),
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int16_t>::max() - 1,
         std::numeric_limits<int16_t>::max()},
        {std::numeric_limits<int16_t>::max(),
         std::nullopt,
         std::numeric_limits<int16_t>::max() - 1}}},
      // Adjacent to min with nulls (min, min+1)
      {{{std::numeric_limits<int16_t>::min(),
         std::numeric_limits<int16_t>::min() + 1,
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int16_t>::min(),
         std::numeric_limits<int16_t>::min() + 1},
        {std::numeric_limits<int16_t>::min() + 1,
         std::nullopt,
         std::numeric_limits<int16_t>::min()}}},
      // Boundary transitions with nulls (min, null, 0, null, max)
      {{{std::numeric_limits<int16_t>::min(),
         std::nullopt,
         0,
         std::nullopt,
         std::numeric_limits<int16_t>::max()},
        {std::nullopt,
         std::numeric_limits<int16_t>::min(),
         std::nullopt,
         std::numeric_limits<int16_t>::max(),
         std::nullopt},
        {0, 0, 0, 0, 0}}},
      // Duplicate max values with nulls
      {{{std::numeric_limits<int16_t>::max(),
         std::numeric_limits<int16_t>::max(),
         std::nullopt,
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int16_t>::max(),
         std::numeric_limits<int16_t>::max(),
         std::nullopt},
        {std::numeric_limits<int16_t>::max(),
         std::nullopt,
         std::nullopt,
         std::numeric_limits<int16_t>::max()}}},
      // Duplicate min values with nulls
      {{{std::numeric_limits<int16_t>::min(),
         std::numeric_limits<int16_t>::min(),
         std::nullopt,
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int16_t>::min(),
         std::numeric_limits<int16_t>::min(),
         std::nullopt},
        {std::numeric_limits<int16_t>::min(),
         std::nullopt,
         std::nullopt,
         std::numeric_limits<int16_t>::min()}}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ASSERT_EQ(testCase.columnValues.size(), 3);
    const auto input = makeRowVector(
        {makeNullableFlatVector<int16_t>(testCase.columnValues[0]),
         makeNullableFlatVector<int16_t>(testCase.columnValues[1]),
         makeNullableFlatVector<int16_t>(testCase.columnValues[2])});
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};

    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, byteTypeWithoutNulls) {
  struct {
    std::vector<std::vector<int8_t>> columnValues;

    std::string debugString() const {
      const size_t numRows = columnValues[0].size();
      std::vector<std::string> rows;
      rows.reserve(numRows);

      for (size_t row = 0; row < numRows; ++row) {
        std::vector<int> rowValues;
        rowValues.reserve(columnValues.size());
        for (const auto& column : columnValues) {
          rowValues.push_back(static_cast<int>(column[row]));
        }
        rows.push_back(fmt::format("{{{}}}", fmt::join(rowValues, ", ")));
      }

      return fmt::format("[{}]", fmt::join(rows, ", "));
    }
  } testCases[] = {
      // Single row
      {{{1}, {2}, {3}}},
      // All same values
      {{{5, 5, 5}, {10, 10, 10}, {15, 15, 15}}},
      // Ascending order
      {{{1, 2, 3, 4}, {100, 100, 100, 100}, {50, 50, 50, 50}}},
      // Descending order
      {{{10, 9, 8, 7}, {0, 0, 0, 0}, {5, 5, 5, 5}}},
      // Mixed positive and negative values
      {{{-5, 0, 5}, {-10, -5, 0}, {10, 5, 0}}},
      // Edge values: min and max int8_t
      {{{std::numeric_limits<int8_t>::min(),
         std::numeric_limits<int8_t>::max(),
         0},
        {-1, 0, 1},
        {100, 50, 25}}},
      // All zeros
      {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
      // Duplicate values
      {{{1, 2, 1, 2}, {3, 3, 4, 4}, {5, 5, 5, 5}}},
      // Large spread (within int8 range)
      {{{-120, -1, 0, 1, 120}, {100, 50, 0, -50, -100}, {42, 42, 42, 42, 42}}},
      // Values differing only in last column
      {{{1, 1, 1}, {2, 2, 2}, {3, 4, 5}}},
      // Values differing only in first column
      {{{1, 2, 3}, {100, 100, 100}, {50, 50, 50}}},
      // Alternating pattern
      {{{1, -1, 1, -1}, {2, -2, 2, -2}, {3, -3, 3, -3}}},
      // All max values (127)
      {{{std::numeric_limits<int8_t>::max(),
         std::numeric_limits<int8_t>::max()},
        {std::numeric_limits<int8_t>::max(),
         std::numeric_limits<int8_t>::max()},
        {std::numeric_limits<int8_t>::max(),
         std::numeric_limits<int8_t>::max()}}},
      // All min values (-128)
      {{{std::numeric_limits<int8_t>::min(),
         std::numeric_limits<int8_t>::min()},
        {std::numeric_limits<int8_t>::min(),
         std::numeric_limits<int8_t>::min()},
        {std::numeric_limits<int8_t>::min(),
         std::numeric_limits<int8_t>::min()}}},
      // Mix of max and min
      {{{std::numeric_limits<int8_t>::max(),
         std::numeric_limits<int8_t>::min()},
        {std::numeric_limits<int8_t>::min(),
         std::numeric_limits<int8_t>::max()},
        {0, 0}}},
      // Adjacent to max (max-1, max)
      {{{std::numeric_limits<int8_t>::max() - 1,
         std::numeric_limits<int8_t>::max()},
        {std::numeric_limits<int8_t>::max(),
         std::numeric_limits<int8_t>::max() - 1},
        {0, 1}}},
      // Adjacent to min (min, min+1)
      {{{std::numeric_limits<int8_t>::min(),
         std::numeric_limits<int8_t>::min() + 1},
        {std::numeric_limits<int8_t>::min() + 1,
         std::numeric_limits<int8_t>::min()},
        {0, 1}}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ASSERT_EQ(testCase.columnValues.size(), 3);
    const auto input = makeRowVector(
        {makeFlatVector<int8_t>(testCase.columnValues[0]),
         makeFlatVector<int8_t>(testCase.columnValues[1]),
         makeFlatVector<int8_t>(testCase.columnValues[2])});
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};

    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, byteTypeWithNulls) {
  struct {
    std::vector<std::vector<std::optional<int8_t>>> columnValues;

    std::string debugString() const {
      const size_t numRows = columnValues[0].size();
      std::vector<std::string> rows;
      rows.reserve(numRows);

      for (size_t row = 0; row < numRows; ++row) {
        std::vector<std::string> rowValues;
        rowValues.reserve(columnValues.size());
        for (const auto& column : columnValues) {
          if (column[row].has_value()) {
            rowValues.push_back(
                std::to_string(static_cast<int>(column[row].value())));
          } else {
            rowValues.push_back("null");
          }
        }
        rows.push_back(fmt::format("{{{}}}", fmt::join(rowValues, ", ")));
      }

      return fmt::format("[{}]", fmt::join(rows, ", "));
    }
  } testCases[] = {
      // Single row with null
      {{{std::nullopt}, {1}, {2}}},
      // Single row without null (nullable type)
      {{{1}, {2}, {3}}},
      // All nulls
      {{{std::nullopt, std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt, std::nullopt}}},
      // Null at beginning with max value
      {{{std::nullopt, 1, std::numeric_limits<int8_t>::max()},
        {10, 20, 30},
        {100, 50, 25}}},
      // Null at end with min value
      {{{1, std::numeric_limits<int8_t>::min(), std::nullopt},
        {10, 20, 30},
        {100, 50, 25}}},
      // Null in middle
      {{{1, std::nullopt, 2}, {10, 20, 30}, {100, 50, 25}}},
      // Nulls in all positions across different columns with edge values
      {{{std::nullopt, 1, std::numeric_limits<int8_t>::max(), 3},
        {1, std::nullopt, 2, std::numeric_limits<int8_t>::min()},
        {std::numeric_limits<int8_t>::min(), 2, std::nullopt, 3}}},
      // Multiple nulls in same row
      {{{std::nullopt, 1, 2}, {std::nullopt, 1, 2}, {std::nullopt, 1, 2}}},
      // Alternating null and non-null with boundaries
      {{{std::nullopt,
         std::numeric_limits<int8_t>::max(),
         std::nullopt,
         2,
         std::nullopt,
         std::numeric_limits<int8_t>::min()},
        {10, 10, 10, 10, 10, 10},
        {20, 20, 20, 20, 20, 20}}},
      // Same values, different null positions
      {{{1, 1, std::nullopt}, {2, std::nullopt, 2}, {3, 3, 3}}},
      // Mixed edge values with nulls
      {{{std::nullopt,
         std::numeric_limits<int8_t>::min(),
         std::numeric_limits<int8_t>::max()},
        {0, std::nullopt, 0},
        {100, 50, std::nullopt}}},
      // Nulls with zeros and boundaries
      {{{std::nullopt, 0, std::numeric_limits<int8_t>::max()},
        {0, std::nullopt, std::numeric_limits<int8_t>::min()},
        {0, 0, 0}}},
      // Nulls with positive and negative including extremes
      {{{std::nullopt, -5, std::numeric_limits<int8_t>::max()},
        {std::numeric_limits<int8_t>::min(), std::nullopt, 10},
        {std::nullopt, -15, std::nullopt}}},
      // Duplicate non-null values with nulls
      {{{1, 1, std::nullopt, std::nullopt},
        {2, 2, 2, std::nullopt},
        {3, std::nullopt, 3, 3}}},
      // Ascending order with nulls and max values
      {{{std::nullopt, 1, 2, std::numeric_limits<int8_t>::max(), std::nullopt},
        {10, std::nullopt, 20, std::nullopt, 30},
        {std::nullopt, std::nullopt, 100, 50, 25}}},
      // Descending order with nulls and min values
      {{{std::nullopt,
         std::numeric_limits<int8_t>::max(),
         2,
         std::numeric_limits<int8_t>::min(),
         std::nullopt},
        {30, std::nullopt, 20, std::nullopt, 10},
        {std::nullopt, std::nullopt, 100, 50, 25}}},
      // Only first column has nulls with boundaries
      {{{std::nullopt,
         std::nullopt,
         std::numeric_limits<int8_t>::min(),
         std::numeric_limits<int8_t>::max()},
        {10, 20, 30, 40},
        {5, 6, 7, 8}}},
      // Only middle column has nulls
      {{{1, 2, 3, 4}, {std::nullopt, std::nullopt, 10, 20}, {100, 50, 25, 10}}},
      // Only last column has nulls with max
      {{{1, 2, 3, std::numeric_limits<int8_t>::max()},
        {10, 20, 30, 40},
        {std::nullopt, std::nullopt, 100, std::numeric_limits<int8_t>::min()}}},
      // All columns have at least one null mixed with extremes
      {{{std::nullopt, std::numeric_limits<int8_t>::max(), 2, 3, 4},
        {1, std::nullopt, std::numeric_limits<int8_t>::min(), 3, 4},
        {1, 2, std::nullopt, std::numeric_limits<int8_t>::max(), 4}}},
      // Null vs non-null with same other column values
      {{{std::nullopt, 1}, {100, 100}, {50, 50}}},
      // Large spread with nulls and boundaries (within int8 range)
      {{{std::nullopt,
         -120,
         std::numeric_limits<int8_t>::min(),
         std::numeric_limits<int8_t>::max(),
         std::nullopt},
        {100, std::nullopt, 0, std::nullopt, -100},
        {std::nullopt, 42, std::nullopt, 42, std::nullopt}}},
      // Mix of nulls and regular values testing sort order
      {{{std::nullopt, std::nullopt, 0, 0},
        {std::nullopt, 1, std::nullopt, 1},
        {2, std::nullopt, 2, std::nullopt}}},
      // Max and min together in different positions
      {{{std::numeric_limits<int8_t>::max(),
         std::numeric_limits<int8_t>::min(),
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int8_t>::max(),
         std::numeric_limits<int8_t>::min()},
        {std::numeric_limits<int8_t>::min(),
         std::nullopt,
         std::numeric_limits<int8_t>::max()}}},
      // Adjacent to max with nulls (max-1, max)
      {{{std::numeric_limits<int8_t>::max() - 1,
         std::numeric_limits<int8_t>::max(),
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int8_t>::max() - 1,
         std::numeric_limits<int8_t>::max()},
        {std::numeric_limits<int8_t>::max(),
         std::nullopt,
         std::numeric_limits<int8_t>::max() - 1}}},
      // Adjacent to min with nulls (min, min+1)
      {{{std::numeric_limits<int8_t>::min(),
         std::numeric_limits<int8_t>::min() + 1,
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int8_t>::min(),
         std::numeric_limits<int8_t>::min() + 1},
        {std::numeric_limits<int8_t>::min() + 1,
         std::nullopt,
         std::numeric_limits<int8_t>::min()}}},
      // Boundary transitions with nulls (min, null, 0, null, max)
      {{{std::numeric_limits<int8_t>::min(),
         std::nullopt,
         0,
         std::nullopt,
         std::numeric_limits<int8_t>::max()},
        {std::nullopt,
         std::numeric_limits<int8_t>::min(),
         std::nullopt,
         std::numeric_limits<int8_t>::max(),
         std::nullopt},
        {0, 0, 0, 0, 0}}},
      // Duplicate max values with nulls
      {{{std::numeric_limits<int8_t>::max(),
         std::numeric_limits<int8_t>::max(),
         std::nullopt,
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int8_t>::max(),
         std::numeric_limits<int8_t>::max(),
         std::nullopt},
        {std::numeric_limits<int8_t>::max(),
         std::nullopt,
         std::nullopt,
         std::numeric_limits<int8_t>::max()}}},
      // Duplicate min values with nulls
      {{{std::numeric_limits<int8_t>::min(),
         std::numeric_limits<int8_t>::min(),
         std::nullopt,
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<int8_t>::min(),
         std::numeric_limits<int8_t>::min(),
         std::nullopt},
        {std::numeric_limits<int8_t>::min(),
         std::nullopt,
         std::nullopt,
         std::numeric_limits<int8_t>::min()}}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ASSERT_EQ(testCase.columnValues.size(), 3);
    const auto input = makeRowVector(
        {makeNullableFlatVector<int8_t>(testCase.columnValues[0]),
         makeNullableFlatVector<int8_t>(testCase.columnValues[1]),
         makeNullableFlatVector<int8_t>(testCase.columnValues[2])});
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};

    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, doubleTypeWithoutNulls) {
  struct {
    std::vector<std::vector<double>> columnValues;

    std::string debugString() const {
      const size_t numRows = columnValues[0].size();
      std::vector<std::string> rows;
      rows.reserve(numRows);

      for (size_t row = 0; row < numRows; ++row) {
        std::vector<std::string> rowValues;
        rowValues.reserve(columnValues.size());
        for (const auto& column : columnValues) {
          rowValues.push_back(std::to_string(column[row]));
        }
        rows.push_back(fmt::format("{{{}}}", fmt::join(rowValues, ", ")));
      }

      return fmt::format("[{}]", fmt::join(rows, ", "));
    }
  } testCases[] = {
      // Single row
      {{{1.0}, {2.0}, {3.0}}},
      // All same values
      {{{5.5, 5.5, 5.5}, {10.5, 10.5, 10.5}, {15.5, 15.5, 15.5}}},
      // Ascending order
      {{{1.1, 2.2, 3.3, 4.4},
        {100.5, 100.5, 100.5, 100.5},
        {50.5, 50.5, 50.5, 50.5}}},
      // Descending order
      {{{10.9, 9.8, 8.7, 7.6}, {0.0, 0.0, 0.0, 0.0}, {5.5, 5.5, 5.5, 5.5}}},
      // Mixed positive and negative values
      {{{-5.5, 0.0, 5.5}, {-10.5, -5.5, 0.0}, {10.5, 5.5, 0.0}}},
      // All zeros
      {{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}},
      // Positive and negative zero
      {{{0.0, -0.0, 0.0}, {-0.0, 0.0, -0.0}, {0.0, 0.0, 0.0}}},
      // Positive infinity
      {{{std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity()},
        {1.0, 2.0},
        {3.0, 4.0}}},
      // Negative infinity
      {{{-std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity()},
        {1.0, 2.0},
        {3.0, 4.0}}},
      // Mix of infinity and regular values
      {{{-std::numeric_limits<double>::infinity(),
         0.0,
         std::numeric_limits<double>::infinity()},
        {-1.0, 0.0, 1.0},
        {100.0, 200.0, 300.0}}},
      // Both infinities together
      {{{-std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity()},
        {std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity()},
        {0.0, 0.0}}},
      // Edge values with infinity
      {{{-std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::lowest(),
         std::numeric_limits<double>::min(),
         0.0,
         std::numeric_limits<double>::max(),
         std::numeric_limits<double>::infinity()},
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
        {10.0, 20.0, 30.0, 40.0, 50.0, 60.0}}},
      // All positive infinity
      {{{std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity()},
        {std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity()},
        {std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity()}}},
      // All negative infinity
      {{{-std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity()},
        {-std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity()},
        {-std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity()}}},
      // Max values
      {{{std::numeric_limits<double>::max(),
         std::numeric_limits<double>::max()},
        {std::numeric_limits<double>::max(),
         std::numeric_limits<double>::max()},
        {std::numeric_limits<double>::max(),
         std::numeric_limits<double>::max()}}},
      // Min positive values
      {{{std::numeric_limits<double>::min(),
         std::numeric_limits<double>::min()},
        {std::numeric_limits<double>::min(),
         std::numeric_limits<double>::min()},
        {std::numeric_limits<double>::min(),
         std::numeric_limits<double>::min()}}},
      // Lowest values (most negative)
      {{{std::numeric_limits<double>::lowest(),
         std::numeric_limits<double>::lowest()},
        {std::numeric_limits<double>::lowest(),
         std::numeric_limits<double>::lowest()},
        {std::numeric_limits<double>::lowest(),
         std::numeric_limits<double>::lowest()}}},
      // Mix of max, min, and infinity
      {{{-std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::lowest(),
         std::numeric_limits<double>::max(),
         std::numeric_limits<double>::infinity()},
        {std::numeric_limits<double>::min(), 0.0, 1.0, 2.0},
        {-1.0, -2.0, -3.0, -4.0}}},
      // Very small and very large values
      {{{-1e308, -1e100, -1e-308, 0.0, 1e-308, 1e100, 1e308},
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0}}},
      // Fractional values
      {{{0.1, 0.01, 0.001, 0.0001},
        {-0.1, -0.01, -0.001, -0.0001},
        {1.5, 2.5, 3.5, 4.5}}},
      // Values differing only in last column
      {{{1.5, 1.5, 1.5}, {2.5, 2.5, 2.5}, {3.5, 4.5, 5.5}}},
      // Values differing only in first column
      {{{1.5, 2.5, 3.5}, {100.5, 100.5, 100.5}, {200.5, 200.5, 200.5}}},
      {{{1.0f, 0.0f, -0.0f}, {0.0f, 1.0f, 0.0f}, {-0.0f, 0.0f, 1.0f}}}};

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ASSERT_EQ(testCase.columnValues.size(), 3);
    const auto input = makeRowVector(
        {makeFlatVector<double>(testCase.columnValues[0]),
         makeFlatVector<double>(testCase.columnValues[1]),
         makeFlatVector<double>(testCase.columnValues[2])});
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};

    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, doubleTypeWithNulls) {
  struct {
    std::vector<std::vector<std::optional<double>>> columnValues;

    std::string debugString() const {
      const size_t numRows = columnValues[0].size();
      std::vector<std::string> rows;
      rows.reserve(numRows);

      for (size_t row = 0; row < numRows; ++row) {
        std::vector<std::string> rowValues;
        rowValues.reserve(columnValues.size());
        for (const auto& column : columnValues) {
          if (column[row].has_value()) {
            rowValues.push_back(std::to_string(column[row].value()));
          } else {
            rowValues.push_back("null");
          }
        }
        rows.push_back(fmt::format("{{{}}}", fmt::join(rowValues, ", ")));
      }

      return fmt::format("[{}]", fmt::join(rows, ", "));
    }
  } testCases[] = {
      // Single row with null
      {{{std::nullopt}, {1.0}, {2.0}}},
      // Single row without null (nullable type)
      {{{1.0}, {2.0}, {3.0}}},
      // All nulls
      {{{std::nullopt, std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt, std::nullopt}}},
      // Null at beginning with infinity
      {{{std::nullopt, 1.0, std::numeric_limits<double>::infinity()},
        {10.0, 20.0, 30.0},
        {100.0, 200.0, 300.0}}},
      // Null at end with negative infinity
      {{{1.0, -std::numeric_limits<double>::infinity(), std::nullopt},
        {10.0, 20.0, 30.0},
        {100.0, 200.0, 300.0}}},
      // Null in middle
      {{{1.0, std::nullopt, 2.0}, {10.0, 20.0, 30.0}, {100.0, 200.0, 300.0}}},
      // Nulls with both infinities
      {{{std::nullopt,
         -std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity()},
        {std::numeric_limits<double>::infinity(),
         std::nullopt,
         -std::numeric_limits<double>::infinity()},
        {0.0, 1.0, std::nullopt}}},
      // Multiple nulls in same row
      {{{std::nullopt, 1.0, 2.0},
        {std::nullopt, 1.0, 2.0},
        {std::nullopt, 1.0, 2.0}}},
      // Alternating null and non-null with infinities
      {{{std::nullopt,
         std::numeric_limits<double>::infinity(),
         std::nullopt,
         2.0,
         std::nullopt,
         -std::numeric_limits<double>::infinity()},
        {10.0, 10.0, 10.0, 10.0, 10.0, 10.0},
        {20.0, 20.0, 20.0, 20.0, 20.0, 20.0}}},
      // Same values, different null positions
      {{{1.5, 1.5, std::nullopt}, {2.5, std::nullopt, 2.5}, {3.5, 3.5, 3.5}}},
      // Nulls with zeros
      {{{std::nullopt, 0.0, std::nullopt},
        {0.0, std::nullopt, 0.0},
        {0.0, 0.0, 0.0}}},
      // Nulls with positive and negative values
      {{{std::nullopt, -5.5, 5.5},
        {-10.5, std::nullopt, 10.5},
        {std::nullopt, -15.5, std::nullopt}}},
      // Duplicate non-null values with nulls
      {{{1.5, 1.5, std::nullopt, std::nullopt},
        {2.5, 2.5, 2.5, std::nullopt},
        {3.5, std::nullopt, 3.5, 3.5}}},
      // Ascending order with nulls and infinity
      {{{std::nullopt,
         1.0,
         2.0,
         std::numeric_limits<double>::infinity(),
         std::nullopt},
        {10.0, std::nullopt, 20.0, std::nullopt, 30.0},
        {std::nullopt, std::nullopt, 100.0, 200.0, 300.0}}},
      // Descending order with nulls and negative infinity
      {{{std::nullopt,
         std::numeric_limits<double>::infinity(),
         2.0,
         -std::numeric_limits<double>::infinity(),
         std::nullopt},
        {30.0, std::nullopt, 20.0, std::nullopt, 10.0},
        {std::nullopt, std::nullopt, 300.0, 200.0, 100.0}}},
      // Only first column has nulls with infinities
      {{{std::nullopt,
         std::nullopt,
         -std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity()},
        {10.0, 20.0, 30.0, 40.0},
        {5.0, 6.0, 7.0, 8.0}}},
      // Only middle column has nulls
      {{{1.0, 2.0, 3.0, 4.0},
        {std::nullopt, std::nullopt, 10.0, 20.0},
        {100.0, 200.0, 300.0, 400.0}}},
      // Only last column has nulls with infinity
      {{{1.0, 2.0, 3.0, std::numeric_limits<double>::infinity()},
        {10.0, 20.0, 30.0, 40.0},
        {std::nullopt,
         std::nullopt,
         100.0,
         -std::numeric_limits<double>::infinity()}}},
      // All columns have at least one null mixed with infinities
      {{{std::nullopt, std::numeric_limits<double>::infinity(), 2.0, 3.0, 4.0},
        {1.0, std::nullopt, -std::numeric_limits<double>::infinity(), 3.0, 4.0},
        {1.0,
         2.0,
         std::nullopt,
         std::numeric_limits<double>::infinity(),
         4.0}}},
      // Null vs non-null with same other column values
      {{{std::nullopt, 1.0}, {100.0, 100.0}, {200.0, 200.0}}},
      // Mix of null, infinity, and regular values
      {{{std::nullopt,
         -std::numeric_limits<double>::infinity(),
         0.0,
         std::numeric_limits<double>::infinity(),
         std::nullopt},
        {std::numeric_limits<double>::infinity(),
         std::nullopt,
         0.0,
         std::nullopt,
         -std::numeric_limits<double>::infinity()},
        {std::nullopt, 1.0, std::nullopt, 2.0, std::nullopt}}},
      // All positive infinity with nulls
      {{{std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(),
         std::nullopt,
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::infinity(),
         std::nullopt},
        {std::numeric_limits<double>::infinity(),
         std::nullopt,
         std::nullopt,
         std::numeric_limits<double>::infinity()}}},
      // All negative infinity with nulls
      {{{-std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity(),
         std::nullopt,
         std::nullopt},
        {std::nullopt,
         -std::numeric_limits<double>::infinity(),
         -std::numeric_limits<double>::infinity(),
         std::nullopt},
        {-std::numeric_limits<double>::infinity(),
         std::nullopt,
         std::nullopt,
         -std::numeric_limits<double>::infinity()}}},
      // Max values with nulls
      {{{std::nullopt, std::numeric_limits<double>::max(), std::nullopt},
        {std::numeric_limits<double>::max(),
         std::nullopt,
         std::numeric_limits<double>::max()},
        {std::nullopt, std::nullopt, std::numeric_limits<double>::max()}}},
      // Lowest values with nulls
      {{{std::nullopt, std::numeric_limits<double>::lowest(), std::nullopt},
        {std::numeric_limits<double>::lowest(),
         std::nullopt,
         std::numeric_limits<double>::lowest()},
        {std::nullopt, std::nullopt, std::numeric_limits<double>::lowest()}}},
      // Mix of max, lowest, infinity and nulls
      {{{std::nullopt,
         std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::max()},
        {std::numeric_limits<double>::lowest(),
         std::nullopt,
         -std::numeric_limits<double>::infinity()},
        {0.0, 1.0, std::nullopt}}},
      // Very small and very large values with nulls
      {{{std::nullopt, 1e-308, 1e308, std::nullopt},
        {1e308, std::nullopt, -1e308, -1e-308},
        {std::nullopt, 0.0f, std::nullopt, 0.0f}}},
      // Positive and negative zero with nulls
      {{{std::nullopt, 0.0, -0.0f},
        {0.0f, std::nullopt, 0.0f},
        {-0.0f, 0.0f, std::nullopt}}}};

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ASSERT_EQ(testCase.columnValues.size(), 3);
    const auto input = makeRowVector(
        {makeNullableFlatVector<double>(testCase.columnValues[0]),
         makeNullableFlatVector<double>(testCase.columnValues[1]),
         makeNullableFlatVector<double>(testCase.columnValues[2])});
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};

    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, floatTypeWithoutNulls) {
  struct {
    std::vector<std::vector<float>> columnValues;

    std::string debugString() const {
      const size_t numRows = columnValues[0].size();
      std::vector<std::string> rows;
      rows.reserve(numRows);

      for (size_t row = 0; row < numRows; ++row) {
        std::vector<std::string> rowValues;
        rowValues.reserve(columnValues.size());
        for (const auto& column : columnValues) {
          rowValues.push_back(std::to_string(column[row]));
        }
        rows.push_back(fmt::format("{{{}}}", fmt::join(rowValues, ", ")));
      }

      return fmt::format("[{}]", fmt::join(rows, ", "));
    }
  } testCases[] = {
      // Single row
      {{{1.0f}, {2.0f}, {3.0f}}},
      // All same values
      {{{5.5f, 5.5f, 5.5f}, {10.5f, 10.5f, 10.5f}, {15.5f, 15.5f, 15.5f}}},
      // Ascending order
      {{{1.1f, 2.2f, 3.3f, 4.4f},
        {100.5f, 100.5f, 100.5f, 100.5f},
        {50.5f, 50.5f, 50.5f, 50.5f}}},
      // Descending order
      {{{10.9f, 9.8f, 8.7f, 7.6f},
        {0.0f, 0.0f, 0.0f, 0.0f},
        {5.5f, 5.5f, 5.5f, 5.5f}}},
      // Mixed positive and negative values
      {{{-5.5f, 0.0f, 5.5f}, {-10.5f, -5.5f, 0.0f}, {10.5f, 5.5f, 0.0f}}},
      // All zeros
      {{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}}},
      // Positive and negative zero
      {{{0.0f, -0.0f, 0.0f}, {-0.0f, 0.0f, -0.0f}, {0.0f, 0.0f, 0.0f}}},
      // Positive infinity
      {{{std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity()},
        {1.0f, 2.0f},
        {3.0f, 4.0f}}},
      // Negative infinity
      {{{-std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity()},
        {1.0f, 2.0f},
        {3.0f, 4.0f}}},
      // Mix of infinity and regular values
      {{{-std::numeric_limits<float>::infinity(),
         0.0f,
         std::numeric_limits<float>::infinity()},
        {-1.0f, 0.0f, 1.0f},
        {100.0f, 200.0f, 300.0f}}},
      // Both infinities together
      {{{-std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity()},
        {std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity()},
        {0.0f, 0.0f}}},
      // Edge values with infinity
      {{{-std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::lowest(),
         std::numeric_limits<float>::min(),
         0.0f,
         std::numeric_limits<float>::max(),
         std::numeric_limits<float>::infinity()},
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}}},
      // All positive infinity
      {{{std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity()},
        {std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity()},
        {std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity()}}},
      // All negative infinity
      {{{-std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity()},
        {-std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity()},
        {-std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity()}}},
      // Max values
      {{{std::numeric_limits<float>::max(), std::numeric_limits<float>::max()},
        {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()},
        {std::numeric_limits<float>::max(),
         std::numeric_limits<float>::max()}}},
      // Min positive values
      {{{std::numeric_limits<float>::min(), std::numeric_limits<float>::min()},
        {std::numeric_limits<float>::min(), std::numeric_limits<float>::min()},
        {std::numeric_limits<float>::min(),
         std::numeric_limits<float>::min()}}},
      // Lowest values (most negative)
      {{{std::numeric_limits<float>::lowest(),
         std::numeric_limits<float>::lowest()},
        {std::numeric_limits<float>::lowest(),
         std::numeric_limits<float>::lowest()},
        {std::numeric_limits<float>::lowest(),
         std::numeric_limits<float>::lowest()}}},
      // Mix of max, min, and infinity
      {{{-std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::lowest(),
         std::numeric_limits<float>::max(),
         std::numeric_limits<float>::infinity()},
        {std::numeric_limits<float>::min(), 0.0f, 1.0f, 2.0f},
        {-1.0f, -2.0f, -3.0f, -4.0f}}},
      // Very small and very large values
      {{{-1e38f, -1e10f, -1e-38f, 0.0f, 1e-38f, 1e10f, 1e38f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}}},
      // Fractional values
      {{{0.1f, 0.01f, 0.001f, 0.0001f},
        {-0.1f, -0.01f, -0.001f, -0.0001f},
        {1.5f, 2.5f, 3.5f, 4.5f}}},
      // Values differing only in last column
      {{{1.5f, 1.5f, 1.5f}, {2.5f, 2.5f, 2.5f}, {3.5f, 4.5f, 5.5f}}},
      // Values differing only in first column
      {{{1.5f, 2.5f, 3.5f},
        {100.5f, 100.5f, 100.5f},
        {200.5f, 200.5f, 200.5f}}},
      {{{1.0f, 0.0f, -0.0f}, {0.0f, 1.0f, 0.0f}, {-0.0f, 0.0f, 1.0f}}}};

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ASSERT_EQ(testCase.columnValues.size(), 3);
    const auto input = makeRowVector(
        {makeFlatVector<float>(testCase.columnValues[0]),
         makeFlatVector<float>(testCase.columnValues[1]),
         makeFlatVector<float>(testCase.columnValues[2])});
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};

    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, floatTypeWithNulls) {
  struct {
    std::vector<std::vector<std::optional<float>>> columnValues;

    std::string debugString() const {
      const size_t numRows = columnValues[0].size();
      std::vector<std::string> rows;
      rows.reserve(numRows);

      for (size_t row = 0; row < numRows; ++row) {
        std::vector<std::string> rowValues;
        rowValues.reserve(columnValues.size());
        for (const auto& column : columnValues) {
          if (column[row].has_value()) {
            rowValues.push_back(std::to_string(column[row].value()));
          } else {
            rowValues.push_back("null");
          }
        }
        rows.push_back(fmt::format("{{{}}}", fmt::join(rowValues, ", ")));
      }

      return fmt::format("[{}]", fmt::join(rows, ", "));
    }
  } testCases[] = {
      // Single row with null
      {{{std::nullopt}, {1.0f}, {2.0f}}},
      // Single row without null (nullable type)
      {{{1.0f}, {2.0f}, {3.0f}}},
      // All nulls
      {{{std::nullopt, std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt, std::nullopt}}},
      // Null at beginning with infinity
      {{{std::nullopt, 1.0f, std::numeric_limits<float>::infinity()},
        {10.0f, 20.0f, 30.0f},
        {100.0f, 200.0f, 300.0f}}},
      // Null at end with negative infinity
      {{{1.0f, -std::numeric_limits<float>::infinity(), std::nullopt},
        {10.0f, 20.0f, 30.0f},
        {100.0f, 200.0f, 300.0f}}},
      // Null in middle
      {{{1.0f, std::nullopt, 2.0f},
        {10.0f, 20.0f, 30.0f},
        {100.0f, 200.0f, 300.0f}}},
      // Nulls with both infinities
      {{{std::nullopt,
         -std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity()},
        {std::numeric_limits<float>::infinity(),
         std::nullopt,
         -std::numeric_limits<float>::infinity()},
        {0.0f, 1.0f, std::nullopt}}},
      // Multiple nulls in same row
      {{{std::nullopt, 1.0f, 2.0f},
        {std::nullopt, 1.0f, 2.0f},
        {std::nullopt, 1.0f, 2.0f}}},
      // Alternating null and non-null with infinities
      {{{std::nullopt,
         std::numeric_limits<float>::infinity(),
         std::nullopt,
         2.0f,
         std::nullopt,
         -std::numeric_limits<float>::infinity()},
        {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f},
        {20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f}}},
      // Same values, different null positions
      {{{1.5f, 1.5f, std::nullopt},
        {2.5f, std::nullopt, 2.5f},
        {3.5f, 3.5f, 3.5f}}},
      // Nulls with zeros
      {{{std::nullopt, 0.0f, std::nullopt},
        {0.0f, std::nullopt, 0.0f},
        {0.0f, 0.0f, 0.0f}}},
      // Nulls with positive and negative values
      {{{std::nullopt, -5.5f, 5.5f},
        {-10.5f, std::nullopt, 10.5f},
        {std::nullopt, -15.5f, std::nullopt}}},
      // Duplicate non-null values with nulls
      {{{1.5f, 1.5f, std::nullopt, std::nullopt},
        {2.5f, 2.5f, 2.5f, std::nullopt},
        {3.5f, std::nullopt, 3.5f, 3.5f}}},
      // Ascending order with nulls and infinity
      {{{std::nullopt,
         1.0f,
         2.0f,
         std::numeric_limits<float>::infinity(),
         std::nullopt},
        {10.0f, std::nullopt, 20.0f, std::nullopt, 30.0f},
        {std::nullopt, std::nullopt, 100.0f, 200.0f, 300.0f}}},
      // Descending order with nulls and negative infinity
      {{{std::nullopt,
         std::numeric_limits<float>::infinity(),
         2.0f,
         -std::numeric_limits<float>::infinity(),
         std::nullopt},
        {30.0f, std::nullopt, 20.0f, std::nullopt, 10.0f},
        {std::nullopt, std::nullopt, 300.0f, 200.0f, 100.0f}}},
      // Only first column has nulls with infinities
      {{{std::nullopt,
         std::nullopt,
         -std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity()},
        {10.0f, 20.0f, 30.0f, 40.0f},
        {5.0f, 6.0f, 7.0f, 8.0f}}},
      // Only middle column has nulls
      {{{1.0f, 2.0f, 3.0f, 4.0f},
        {std::nullopt, std::nullopt, 10.0f, 20.0f},
        {100.0f, 200.0f, 300.0f, 400.0f}}},
      // Only last column has nulls with infinity
      {{{1.0f, 2.0f, 3.0f, std::numeric_limits<float>::infinity()},
        {10.0f, 20.0f, 30.0f, 40.0f},
        {std::nullopt,
         std::nullopt,
         100.0f,
         -std::numeric_limits<float>::infinity()}}},
      // All columns have at least one null mixed with infinities
      {{{std::nullopt,
         std::numeric_limits<float>::infinity(),
         2.0f,
         3.0f,
         4.0f},
        {1.0f,
         std::nullopt,
         -std::numeric_limits<float>::infinity(),
         3.0f,
         4.0f},
        {1.0f,
         2.0f,
         std::nullopt,
         std::numeric_limits<float>::infinity(),
         4.0f}}},
      // Null vs non-null with same other column values
      {{{std::nullopt, 1.0f}, {100.0f, 100.0f}, {200.0f, 200.0f}}},
      // Mix of null, infinity, and regular values
      {{{std::nullopt,
         -std::numeric_limits<float>::infinity(),
         0.0f,
         std::numeric_limits<float>::infinity(),
         std::nullopt},
        {std::numeric_limits<float>::infinity(),
         std::nullopt,
         0.0f,
         std::nullopt,
         -std::numeric_limits<float>::infinity()},
        {std::nullopt, 1.0f, std::nullopt, 2.0f, std::nullopt}}},
      // All positive infinity with nulls
      {{{std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity(),
         std::nullopt,
         std::nullopt},
        {std::nullopt,
         std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::infinity(),
         std::nullopt},
        {std::numeric_limits<float>::infinity(),
         std::nullopt,
         std::nullopt,
         std::numeric_limits<float>::infinity()}}},
      // All negative infinity with nulls
      {{{-std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity(),
         std::nullopt,
         std::nullopt},
        {std::nullopt,
         -std::numeric_limits<float>::infinity(),
         -std::numeric_limits<float>::infinity(),
         std::nullopt},
        {-std::numeric_limits<float>::infinity(),
         std::nullopt,
         std::nullopt,
         -std::numeric_limits<float>::infinity()}}},
      // Max values with nulls
      {{{std::nullopt, std::numeric_limits<float>::max(), std::nullopt},
        {std::numeric_limits<float>::max(),
         std::nullopt,
         std::numeric_limits<float>::max()},
        {std::nullopt, std::nullopt, std::numeric_limits<float>::max()}}},
      // Lowest values with nulls
      {{{std::nullopt, std::numeric_limits<float>::lowest(), std::nullopt},
        {std::numeric_limits<float>::lowest(),
         std::nullopt,
         std::numeric_limits<float>::lowest()},
        {std::nullopt, std::nullopt, std::numeric_limits<float>::lowest()}}},
      // Mix of max, lowest, infinity and nulls
      {{{std::nullopt,
         std::numeric_limits<float>::infinity(),
         std::numeric_limits<float>::max()},
        {std::numeric_limits<float>::lowest(),
         std::nullopt,
         -std::numeric_limits<float>::infinity()},
        {0.0f, 1.0f, std::nullopt}}},
      // Very small and very large values with nulls
      {{{std::nullopt, 1e-38f, 1e38f, std::nullopt},
        {1e38f, std::nullopt, -1e38f, -1e-38f},
        {std::nullopt, 0.0f, std::nullopt, 0.0f}}},
      // Positive and negative zero with nulls
      {{{std::nullopt, 0.0f, -0.0f},
        {0.0f, std::nullopt, 0.0f},
        {-0.0f, 0.0f, std::nullopt}}}};

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ASSERT_EQ(testCase.columnValues.size(), 3);
    const auto input = makeRowVector(
        {makeNullableFlatVector<float>(testCase.columnValues[0]),
         makeNullableFlatVector<float>(testCase.columnValues[1]),
         makeNullableFlatVector<float>(testCase.columnValues[2])});
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};

    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, booleanTypeWithoutNulls) {
  struct {
    std::vector<std::vector<bool>> columnValues;

    std::string debugString() const {
      const size_t numRows = columnValues[0].size();
      std::vector<std::string> rows;
      rows.reserve(numRows);

      for (size_t row = 0; row < numRows; ++row) {
        std::vector<std::string> rowValues;
        rowValues.reserve(columnValues.size());
        for (const auto& column : columnValues) {
          rowValues.push_back(column[row] ? "true" : "false");
        }
        rows.push_back(fmt::format("{{{}}}", fmt::join(rowValues, ", ")));
      }

      return fmt::format("[{}]", fmt::join(rows, ", "));
    }
  } testCases[] = {
      // Single row (true)
      {{{true}, {false}, {true}}},
      // Single row (false)
      {{{false}, {true}, {false}}},
      // All true values
      {{{true, true, true}, {true, true, true}, {true, true, true}}},
      // All false values
      {{{false, false, false}, {false, false, false}, {false, false, false}}},
      // Mixed true/false values
      {{{true, false, true, false},
        {false, true, false, true},
        {true, true, false, false}}},
      // Alternating true/false
      {{{true, false, true, false},
        {false, true, false, true},
        {true, false, true, false}}},
      // Values differing only in last column
      {{{true, true, true}, {false, false, false}, {true, false, true}}},
      // Values differing only in first column
      {{{true, false, true}, {false, false, false}, {true, true, true}}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ASSERT_EQ(testCase.columnValues.size(), 3);
    const auto input = makeRowVector(
        {makeFlatVector<bool>(testCase.columnValues[0]),
         makeFlatVector<bool>(testCase.columnValues[1]),
         makeFlatVector<bool>(testCase.columnValues[2])});
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};
    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, booleanTypeWithNulls) {
  struct {
    std::vector<std::vector<std::optional<bool>>> columnValues;

    std::string debugString() const {
      const size_t numRows = columnValues[0].size();
      std::vector<std::string> rows;
      rows.reserve(numRows);

      for (size_t row = 0; row < numRows; ++row) {
        std::vector<std::string> rowValues;
        rowValues.reserve(columnValues.size());
        for (const auto& column : columnValues) {
          if (column[row].has_value()) {
            rowValues.push_back(column[row].value() ? "true" : "false");
          } else {
            rowValues.push_back("null");
          }
        }
        rows.push_back(fmt::format("{{{}}}", fmt::join(rowValues, ", ")));
      }

      return fmt::format("[{}]", fmt::join(rows, ", "));
    }
  } testCases[] = {
      // Single null
      {{{std::nullopt}, {true}, {false}}},
      // All nulls
      {{{std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt}}},
      // Null with true
      {{{std::nullopt, true}, {true, false}, {false, true}}},
      // Null with false
      {{{std::nullopt, false}, {false, true}, {true, false}}},
      // Nulls at different positions
      {{{std::nullopt, true, false},
        {true, std::nullopt, false},
        {true, false, std::nullopt}}},
      // Alternating null/true/false
      {{{std::nullopt, true, false, std::nullopt},
        {true, std::nullopt, true, false},
        {false, true, std::nullopt, true}}},
      // Only first column has nulls
      {{{std::nullopt, std::nullopt, true},
        {true, false, true},
        {false, true, false}}},
      // Only middle column has nulls
      {{{true, false, true},
        {std::nullopt, std::nullopt, true},
        {false, true, false}}},
      // Only last column has nulls
      {{{true, false, true},
        {false, true, false},
        {std::nullopt, std::nullopt, true}}},
      // All columns have at least one null
      {{{std::nullopt, true, false},
        {true, std::nullopt, false},
        {true, false, std::nullopt}}},
      // Mix of null/true/false values
      {{{std::nullopt, true, false, std::nullopt},
        {true, std::nullopt, false, true},
        {false, true, std::nullopt, false}}},
      // Null vs non-null with same other column values
      {{{std::nullopt, true}, {false, false}, {true, true}}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ASSERT_EQ(testCase.columnValues.size(), 3);
    const auto input = makeRowVector(
        {makeNullableFlatVector<bool>(testCase.columnValues[0]),
         makeNullableFlatVector<bool>(testCase.columnValues[1]),
         makeNullableFlatVector<bool>(testCase.columnValues[2])});
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};
    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, stringTypeWithoutNulls) {
  struct {
    std::vector<std::vector<std::string>> columnValues;

    std::string debugString() const {
      const size_t numRows = columnValues[0].size();
      std::vector<std::string> rows;
      rows.reserve(numRows);

      for (size_t row = 0; row < numRows; ++row) {
        std::vector<std::string> rowValues;
        rowValues.reserve(columnValues.size());
        for (const auto& column : columnValues) {
          rowValues.push_back(fmt::format("'{}'", column[row]));
        }
        rows.push_back(fmt::format("{{{}}}", fmt::join(rowValues, ", ")));
      }

      return fmt::format("[{}]", fmt::join(rows, ", "));
    }
  } testCases[] = {
      // Single row
      {{{"apple"}, {"dog"}, {"x"}}},
      // All same values
      {{{"same", "same", "same"},
        {"same", "same", "same"},
        {"same", "same", "same"}}},
      // Ascending order (lexicographically)
      {{{"a", "b", "c", "d"},
        {"alpha", "alpha", "alpha", "alpha"},
        {"x", "x", "x", "x"}}},
      // Descending order (lexicographically)
      {{{"zebra", "yak", "xray", "wolf"},
        {"delta", "delta", "delta", "delta"},
        {"z", "z", "z", "z"}}},
      // Empty string as minimum
      {{{"", "a", "aa", "aaa"},
        {"empty", "empty", "empty", "empty"},
        {"test", "test", "test", "test"}}},
      // Empty strings
      {{{"", "", ""}, {"", "", ""}, {"", "", ""}}},
      // Single character strings
      {{{"a", "b", "c"}, {"x", "y", "z"}, {"1", "2", "3"}}},
      // Multi-character strings
      {{{"apple", "banana", "cherry"},
        {"dog", "elephant", "fox"},
        {"red", "green", "blue"}}},
      // Strings with spaces
      {{{"hello world", "foo bar", "test case"},
        {"space test", "another test", "final test"},
        {"a b c", "x y z", "1 2 3"}}},
      // Strings with special characters
      {{{"hello!", "world?", "test@"},
        {"special#", "chars$", "here%"},
        {"more^", "symbols&", "data*"}}},
      // Strings differing only in case
      {{{"apple", "Apple", "APPLE"},
        {"test", "test", "test"},
        {"value", "value", "value"}}},
      // Strings differing only in length
      {{{"a", "aa", "aaa"}, {"b", "bb", "bbb"}, {"c", "cc", "ccc"}}},
      // Strings differing only in last character
      {{{"testa", "testb", "testc"},
        {"valuex", "valuey", "valuez"},
        {"item1", "item2", "item3"}}},
      // Very long strings
      {{{"this_is_a_very_long_string_that_contains_many_characters_to_test_encoding",
         "another_extremely_long_string_with_different_content_for_testing_purposes",
         "yet_another_long_string_to_ensure_proper_handling_of_large_data"},
        {"long_value_1", "long_value_2", "long_value_3"},
        {"x", "y", "z"}}},
      // Strings with numbers
      {{{"test123", "value456", "item789"},
        {"abc123", "def456", "ghi789"},
        {"1a2b3c", "4d5e6f", "7g8h9i"}}},
      // Strings with unicode characters (if supported)
      {{{"café", "naïve", "résumé"},
        {"hello", "world", "test"},
        {"α", "β", "γ"}}},
      // Duplicate strings
      {{{"apple", "apple", "banana", "banana"},
        {"test", "test", "value", "value"},
        {"x", "y", "x", "y"}}},
      // Prefixes of each other
      {{{"a", "ab", "abc", "abcd"},
        {"test", "test", "test", "test"},
        {"x", "x", "x", "x"}}},
      // Mixed empty and non-empty
      {{{"", "a", "", "b"},
        {"empty", "", "mixed", ""},
        {"test", "test", "", ""}}},
      // Strings with tabs and newlines
      {{{"hello\tworld", "test\ncase", "value\r\ndata"},
        {"tab\there", "new\nline", "return\rchar"},
        {"a", "b", "c"}}},
      // Numeric strings (lexicographic vs numeric order)
      {{{"1", "10", "2", "20"},
        {"100", "200", "300", "400"},
        {"a", "b", "c", "d"}}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ASSERT_EQ(testCase.columnValues.size(), 3);
    const auto input = makeRowVector(
        {makeFlatVector<std::string_view>(std::vector<std::string_view>(
             testCase.columnValues[0].begin(), testCase.columnValues[0].end())),
         makeFlatVector<std::string_view>(std::vector<std::string_view>(
             testCase.columnValues[1].begin(), testCase.columnValues[1].end())),
         makeFlatVector<std::string_view>(std::vector<std::string_view>(
             testCase.columnValues[2].begin(),
             testCase.columnValues[2].end()))});
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};
    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, stringTypeWithNulls) {
  struct {
    std::vector<std::vector<std::optional<std::string>>> columnValues;

    std::string debugString() const {
      const size_t numRows = columnValues[0].size();
      std::vector<std::string> rows;
      rows.reserve(numRows);

      for (size_t row = 0; row < numRows; ++row) {
        std::vector<std::string> rowValues;
        rowValues.reserve(columnValues.size());
        for (const auto& column : columnValues) {
          if (column[row].has_value()) {
            rowValues.push_back(fmt::format("'{}'", column[row].value()));
          } else {
            rowValues.push_back("null");
          }
        }
        rows.push_back(fmt::format("{{{}}}", fmt::join(rowValues, ", ")));
      }

      return fmt::format("[{}]", fmt::join(rows, ", "));
    }
  } testCases[] = {
      // Single null
      {{{std::nullopt}, {"test"}, {"value"}}},
      // All nulls
      {{{std::nullopt, std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt, std::nullopt}}},
      // Null with empty string
      {{{std::nullopt, ""}, {"test", "test"}, {"value", "value"}}},
      // Null with non-empty strings
      {{{std::nullopt, "apple", "banana"},
        {"test", "test", "test"},
        {"x", "y", "z"}}},
      // Nulls at different positions (beginning, middle, end)
      {{{std::nullopt, "a", "b", "c"},
        {"a", std::nullopt, "b", "c"},
        {"a", "b", std::nullopt, "c"}}},
      // Empty string vs null
      {{{std::nullopt, "", "a"}, {"", std::nullopt, "a"}, {"a", "a", "a"}}},
      // Nulls with various string lengths
      {{{std::nullopt, "a", "ab", "abc"},
        {"test", std::nullopt, "testing", "t"},
        {"x", "y", std::nullopt, "z"}}},
      // Duplicate strings with nulls
      {{{std::nullopt, "apple", "apple", std::nullopt},
        {"test", std::nullopt, "test", std::nullopt},
        {"x", "x", std::nullopt, "x"}}},
      // Ascending with nulls
      {{{std::nullopt, "a", "b", "c"},
        {"alpha", std::nullopt, "beta", "gamma"},
        {std::nullopt, "x", "y", "z"}}},
      // Descending with nulls
      {{{std::nullopt, "zebra", "yak", "wolf"},
        {"delta", std::nullopt, "charlie", "bravo"},
        {std::nullopt, "z", "y", "x"}}},
      // Only first column has nulls
      {{{std::nullopt, std::nullopt, "a", "b"},
        {"test1", "test2", "test3", "test4"},
        {"x", "y", "z", "w"}}},
      // Only middle column has nulls
      {{{"apple", "banana", "cherry"},
        {std::nullopt, std::nullopt, "test"},
        {"x", "y", "z"}}},
      // Only last column has nulls
      {{{"apple", "banana", "cherry"},
        {"dog", "elephant", "fox"},
        {std::nullopt, std::nullopt, "red"}}},
      // All columns have at least one null
      {{{std::nullopt, "a", "b", "c"},
        {"test", std::nullopt, "value", "data"},
        {"x", "y", std::nullopt, "z"}}},
      // All columns have nulls in same rows
      {{{std::nullopt, "a", std::nullopt, "c"},
        {std::nullopt, "test", std::nullopt, "value"},
        {std::nullopt, "x", std::nullopt, "z"}}},
      // Null vs non-null with same other values
      {{{std::nullopt, "apple"}, {"test", "test"}, {"value", "value"}}},
      // Mix of empty string, null, and regular strings
      {{{std::nullopt, "", "a", "apple"},
        {"", std::nullopt, "b", "banana"},
        {"test", "test", std::nullopt, ""}}},
      // Nulls with strings containing spaces
      {{{std::nullopt, "hello world", "foo bar"},
        {"test case", std::nullopt, "another test"},
        {std::nullopt, "a b c", std::nullopt}}},
      // Nulls with strings containing special characters
      {{{std::nullopt, "hello!", "world?"},
        {"special#", std::nullopt, "chars$"},
        {"more^", std::nullopt, "data*"}}},
      // Nulls with strings differing only in case
      {{{std::nullopt, "apple", "Apple", "APPLE"},
        {"test", std::nullopt, "test", "test"},
        {"value", "value", std::nullopt, "value"}}},
      // Nulls with strings differing only in length
      {{{std::nullopt, "a", "aa", "aaa"},
        {"b", std::nullopt, "bb", "bbb"},
        {"c", "cc", std::nullopt, "ccc"}}},
      // Nulls with very long strings
      {{{std::nullopt,
         "this_is_a_very_long_string_that_contains_many_characters",
         "another_long_string"},
        {"test", std::nullopt, "value"},
        {"x", "y", std::nullopt}}},
      // Nulls with strings containing numbers
      {{{std::nullopt, "test123", "value456"},
        {"abc123", std::nullopt, "def456"},
        {"1a2b3c", "4d5e6f", std::nullopt}}},
      // Nulls with unicode characters
      {{{std::nullopt, "café", "naïve"},
        {"hello", std::nullopt, "world"},
        {"α", "β", std::nullopt}}},
      // Nulls with prefixes
      {{{std::nullopt, "a", "ab", "abc"},
        {"test", std::nullopt, "test", "test"},
        {"x", "x", std::nullopt, "x"}}},
      // Multiple nulls in same row
      {{{std::nullopt, "a", std::nullopt},
        {std::nullopt, "b", std::nullopt},
        {std::nullopt, "c", std::nullopt}}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ASSERT_EQ(testCase.columnValues.size(), 3);

    // Convert std::optional<std::string> to std::optional<std::string_view>
    std::vector<std::optional<std::string_view>> col0, col1, col2;
    for (const auto& val : testCase.columnValues[0]) {
      col0.push_back(
          val.has_value() ? std::optional<std::string_view>(val.value())
                          : std::nullopt);
    }
    for (const auto& val : testCase.columnValues[1]) {
      col1.push_back(
          val.has_value() ? std::optional<std::string_view>(val.value())
                          : std::nullopt);
    }
    for (const auto& val : testCase.columnValues[2]) {
      col2.push_back(
          val.has_value() ? std::optional<std::string_view>(val.value())
                          : std::nullopt);
    }

    const auto input = makeRowVector(
        {makeNullableFlatVector<std::string_view>(col0),
         makeNullableFlatVector<std::string_view>(col1),
         makeNullableFlatVector<std::string_view>(col2)});
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};

    // Test all four sort orders
    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, timestampTypeWithoutNulls) {
  struct {
    std::vector<std::vector<velox::Timestamp>> columnValues;

    std::string debugString() const {
      const size_t numRows = columnValues[0].size();
      std::vector<std::string> rows;
      rows.reserve(numRows);

      for (size_t row = 0; row < numRows; ++row) {
        std::vector<std::string> rowValues;
        rowValues.reserve(columnValues.size());
        for (const auto& column : columnValues) {
          rowValues.push_back(column[row].toString());
        }
        rows.push_back(fmt::format("{{{}}}", fmt::join(rowValues, ", ")));
      }

      return fmt::format("[{}]", fmt::join(rows, ", "));
    }
  } testCases[] = {
      // Single row
      {{{velox::Timestamp(1, 0)},
        {velox::Timestamp(2, 0)},
        {velox::Timestamp(3, 0)}}},
      // All same values
      {{{velox::Timestamp(5, 100), velox::Timestamp(5, 100)},
        {velox::Timestamp(10, 200), velox::Timestamp(10, 200)},
        {velox::Timestamp(15, 300), velox::Timestamp(15, 300)}}},
      // Ascending order in first column (by seconds)
      {{{velox::Timestamp(1, 0),
         velox::Timestamp(2, 0),
         velox::Timestamp(3, 0)},
        {velox::Timestamp(100, 0),
         velox::Timestamp(100, 0),
         velox::Timestamp(100, 0)},
        {velox::Timestamp(50, 0),
         velox::Timestamp(50, 0),
         velox::Timestamp(50, 0)}}},
      // Descending order in first column (by seconds)
      {{{velox::Timestamp(10, 0),
         velox::Timestamp(9, 0),
         velox::Timestamp(8, 0)},
        {velox::Timestamp(0, 0),
         velox::Timestamp(0, 0),
         velox::Timestamp(0, 0)},
        {velox::Timestamp(5, 0),
         velox::Timestamp(5, 0),
         velox::Timestamp(5, 0)}}},
      // Mixed positive and negative seconds
      {{{velox::Timestamp(-5, 0),
         velox::Timestamp(0, 0),
         velox::Timestamp(5, 0)},
        {velox::Timestamp(-10, 0),
         velox::Timestamp(-5, 0),
         velox::Timestamp(0, 0)},
        {velox::Timestamp(10, 0),
         velox::Timestamp(5, 0),
         velox::Timestamp(0, 0)}}},
      // Same seconds, different nanos
      {{{velox::Timestamp(100, 1),
         velox::Timestamp(100, 2),
         velox::Timestamp(100, 3)},
        {velox::Timestamp(200, 100),
         velox::Timestamp(200, 200),
         velox::Timestamp(200, 300)},
        {velox::Timestamp(300, 999),
         velox::Timestamp(300, 998),
         velox::Timestamp(300, 997)}}},
      // Edge values: valid min and max seconds for Timestamp
      // kMinSeconds = INT64_MIN / 1000 - 1, kMaxSeconds = INT64_MAX / 1000
      {{{velox::Timestamp::min(),
         velox::Timestamp::max(),
         velox::Timestamp(0, 0)},
        {velox::Timestamp(-1, 0),
         velox::Timestamp(0, 0),
         velox::Timestamp(1, 0)},
        {velox::Timestamp(100, 0),
         velox::Timestamp(200, 0),
         velox::Timestamp(300, 0)}}},
      // All zeros
      {{{velox::Timestamp(0, 0),
         velox::Timestamp(0, 0),
         velox::Timestamp(0, 0)},
        {velox::Timestamp(0, 0),
         velox::Timestamp(0, 0),
         velox::Timestamp(0, 0)},
        {velox::Timestamp(0, 0),
         velox::Timestamp(0, 0),
         velox::Timestamp(0, 0)}}},
      // Duplicate values across rows
      {{{velox::Timestamp(1, 100),
         velox::Timestamp(2, 200),
         velox::Timestamp(1, 100),
         velox::Timestamp(2, 200)},
        {velox::Timestamp(3, 300),
         velox::Timestamp(3, 300),
         velox::Timestamp(4, 400),
         velox::Timestamp(4, 400)},
        {velox::Timestamp(5, 500),
         velox::Timestamp(5, 500),
         velox::Timestamp(5, 500),
         velox::Timestamp(5, 500)}}},
      // Large spread of values
      {{{velox::Timestamp(-1000000, 0),
         velox::Timestamp(-1, 0),
         velox::Timestamp(0, 0),
         velox::Timestamp(1, 0),
         velox::Timestamp(1000000, 0)},
        {velox::Timestamp(999999, 0),
         velox::Timestamp(500000, 0),
         velox::Timestamp(0, 0),
         velox::Timestamp(-500000, 0),
         velox::Timestamp(-999999, 0)},
        {velox::Timestamp(42, 0),
         velox::Timestamp(42, 0),
         velox::Timestamp(42, 0),
         velox::Timestamp(42, 0),
         velox::Timestamp(42, 0)}}},
      // Values that differ only in last column
      {{{velox::Timestamp(1, 0),
         velox::Timestamp(1, 0),
         velox::Timestamp(1, 0)},
        {velox::Timestamp(2, 0),
         velox::Timestamp(2, 0),
         velox::Timestamp(2, 0)},
        {velox::Timestamp(3, 0),
         velox::Timestamp(4, 0),
         velox::Timestamp(5, 0)}}},
      // Values that differ only in first column
      {{{velox::Timestamp(1, 0),
         velox::Timestamp(2, 0),
         velox::Timestamp(3, 0)},
        {velox::Timestamp(100, 0),
         velox::Timestamp(100, 0),
         velox::Timestamp(100, 0)},
        {velox::Timestamp(200, 0),
         velox::Timestamp(200, 0),
         velox::Timestamp(200, 0)}}},
      // Nanos ordering within same second
      {{{velox::Timestamp(1, 0),
         velox::Timestamp(1, 500000000),
         velox::Timestamp(1, 999999999)},
        {velox::Timestamp(2, 999999999),
         velox::Timestamp(2, 500000000),
         velox::Timestamp(2, 0)},
        {velox::Timestamp(3, 123456789),
         velox::Timestamp(3, 123456789),
         velox::Timestamp(3, 123456789)}}},
      // All min values
      {{{velox::Timestamp::min(),
         velox::Timestamp::min(),
         velox::Timestamp::min()},
        {velox::Timestamp::min(),
         velox::Timestamp::min(),
         velox::Timestamp::min()},
        {velox::Timestamp::min(),
         velox::Timestamp::min(),
         velox::Timestamp::min()}}},
      // All max values
      {{{velox::Timestamp::max(),
         velox::Timestamp::max(),
         velox::Timestamp::max()},
        {velox::Timestamp::max(),
         velox::Timestamp::max(),
         velox::Timestamp::max()},
        {velox::Timestamp::max(),
         velox::Timestamp::max(),
         velox::Timestamp::max()}}},
      // Min to max range in single column
      {{{velox::Timestamp::min(),
         velox::Timestamp(0, 0),
         velox::Timestamp::max()},
        {velox::Timestamp::min(),
         velox::Timestamp(0, 0),
         velox::Timestamp::max()},
        {velox::Timestamp::min(),
         velox::Timestamp(0, 0),
         velox::Timestamp::max()}}},
      // Min and max with max nanos
      {{{velox::Timestamp(velox::Timestamp::min().getSeconds(), 999999999),
         velox::Timestamp(velox::Timestamp::max().getSeconds(), 999999999)},
        {velox::Timestamp(velox::Timestamp::min().getSeconds(), 0),
         velox::Timestamp(velox::Timestamp::max().getSeconds(), 0)},
        {velox::Timestamp(0, 999999999), velox::Timestamp(0, 0)}}},
      // Alternating min and max
      {{{velox::Timestamp::min(),
         velox::Timestamp::max(),
         velox::Timestamp::min(),
         velox::Timestamp::max()},
        {velox::Timestamp::max(),
         velox::Timestamp::min(),
         velox::Timestamp::max(),
         velox::Timestamp::min()},
        {velox::Timestamp::min(),
         velox::Timestamp::min(),
         velox::Timestamp::max(),
         velox::Timestamp::max()}}}};

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ASSERT_EQ(testCase.columnValues.size(), 3);
    const auto input = makeRowVector(
        {makeFlatVector<velox::Timestamp>(testCase.columnValues[0]),
         makeFlatVector<velox::Timestamp>(testCase.columnValues[1]),
         makeFlatVector<velox::Timestamp>(testCase.columnValues[2])});
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};

    // Test all four sort orders
    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, timestampTypeWithNulls) {
  struct {
    std::vector<std::vector<std::optional<velox::Timestamp>>> columnValues;

    std::string debugString() const {
      const size_t numRows = columnValues[0].size();
      std::vector<std::string> rows;
      rows.reserve(numRows);

      for (size_t row = 0; row < numRows; ++row) {
        std::vector<std::string> rowValues;
        rowValues.reserve(columnValues.size());
        for (const auto& column : columnValues) {
          if (column[row].has_value()) {
            rowValues.push_back(column[row].value().toString());
          } else {
            rowValues.push_back("null");
          }
        }
        rows.push_back(fmt::format("{{{}}}", fmt::join(rowValues, ", ")));
      }

      return fmt::format("[{}]", fmt::join(rows, ", "));
    }
  } testCases[] = {
      // Single row with null
      {{{std::nullopt}, {velox::Timestamp(1, 0)}, {velox::Timestamp(2, 0)}}},
      // Single row without null (nullable type)
      {{{velox::Timestamp(1, 0)},
        {velox::Timestamp(2, 0)},
        {velox::Timestamp(3, 0)}}},
      // All nulls
      {{{std::nullopt, std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt, std::nullopt},
        {std::nullopt, std::nullopt, std::nullopt}}},
      // Null at beginning of first column
      {{{std::nullopt, velox::Timestamp(1, 0), velox::Timestamp(2, 0)},
        {velox::Timestamp(10, 0),
         velox::Timestamp(20, 0),
         velox::Timestamp(30, 0)},
        {velox::Timestamp(100, 0),
         velox::Timestamp(200, 0),
         velox::Timestamp(300, 0)}}},
      // Null at end of first column
      {{{velox::Timestamp(1, 0), velox::Timestamp(2, 0), std::nullopt},
        {velox::Timestamp(10, 0),
         velox::Timestamp(20, 0),
         velox::Timestamp(30, 0)},
        {velox::Timestamp(100, 0),
         velox::Timestamp(200, 0),
         velox::Timestamp(300, 0)}}},
      // Null in middle of first column
      {{{velox::Timestamp(1, 0), std::nullopt, velox::Timestamp(2, 0)},
        {velox::Timestamp(10, 0),
         velox::Timestamp(20, 0),
         velox::Timestamp(30, 0)},
        {velox::Timestamp(100, 0),
         velox::Timestamp(200, 0),
         velox::Timestamp(300, 0)}}},
      // Nulls in all positions across different columns
      {{{std::nullopt,
         velox::Timestamp(1, 0),
         velox::Timestamp(2, 0),
         velox::Timestamp(3, 0)},
        {velox::Timestamp(1, 0),
         std::nullopt,
         velox::Timestamp(2, 0),
         velox::Timestamp(3, 0)},
        {velox::Timestamp(1, 0),
         velox::Timestamp(2, 0),
         std::nullopt,
         velox::Timestamp(3, 0)}}},
      // Multiple nulls in same row
      {{{std::nullopt, velox::Timestamp(1, 0), velox::Timestamp(2, 0)},
        {std::nullopt, velox::Timestamp(10, 0), velox::Timestamp(20, 0)},
        {std::nullopt, velox::Timestamp(100, 0), velox::Timestamp(200, 0)}}},
      // Alternating nulls and values
      {{{std::nullopt,
         velox::Timestamp(1, 0),
         std::nullopt,
         velox::Timestamp(2, 0)},
        {velox::Timestamp(10, 0),
         std::nullopt,
         velox::Timestamp(20, 0),
         std::nullopt},
        {std::nullopt,
         velox::Timestamp(100, 0),
         std::nullopt,
         velox::Timestamp(200, 0)}}},
      // Nulls with valid edge values for Timestamp
      // kMinSeconds = INT64_MIN / 1000 - 1, kMaxSeconds = INT64_MAX / 1000
      {{{std::nullopt, velox::Timestamp::min(), velox::Timestamp::max()},
        {velox::Timestamp(0, 0), std::nullopt, velox::Timestamp(1, 0)},
        {velox::Timestamp(-1, 0), velox::Timestamp(0, 0), std::nullopt}}},
      // Null with varying nanos
      {{{std::nullopt,
         velox::Timestamp(1, 100),
         velox::Timestamp(1, 200),
         velox::Timestamp(1, 300)},
        {velox::Timestamp(2, 100),
         std::nullopt,
         velox::Timestamp(2, 200),
         velox::Timestamp(2, 300)},
        {velox::Timestamp(3, 100),
         velox::Timestamp(3, 200),
         std::nullopt,
         velox::Timestamp(3, 300)}}},
      // Dense nulls at boundaries
      {{{std::nullopt,
         std::nullopt,
         velox::Timestamp(1, 0),
         velox::Timestamp(2, 0),
         std::nullopt,
         std::nullopt},
        {velox::Timestamp(10, 0),
         velox::Timestamp(20, 0),
         std::nullopt,
         std::nullopt,
         velox::Timestamp(30, 0),
         velox::Timestamp(40, 0)},
        {std::nullopt,
         velox::Timestamp(100, 0),
         std::nullopt,
         velox::Timestamp(200, 0),
         std::nullopt,
         velox::Timestamp(300, 0)}}},
      // All min values with some nulls
      {{{velox::Timestamp::min(), std::nullopt, velox::Timestamp::min()},
        {std::nullopt, velox::Timestamp::min(), velox::Timestamp::min()},
        {velox::Timestamp::min(), velox::Timestamp::min(), std::nullopt}}},
      // All max values with some nulls
      {{{velox::Timestamp::max(), std::nullopt, velox::Timestamp::max()},
        {std::nullopt, velox::Timestamp::max(), velox::Timestamp::max()},
        {velox::Timestamp::max(), velox::Timestamp::max(), std::nullopt}}},
      // Min to max range with nulls
      {{{std::nullopt,
         velox::Timestamp::min(),
         velox::Timestamp(0, 0),
         velox::Timestamp::max()},
        {velox::Timestamp::min(),
         std::nullopt,
         velox::Timestamp(0, 0),
         velox::Timestamp::max()},
        {velox::Timestamp::min(),
         velox::Timestamp(0, 0),
         std::nullopt,
         velox::Timestamp::max()}}},
      // Min and max with max nanos and nulls
      {{{std::nullopt,
         velox::Timestamp(velox::Timestamp::min().getSeconds(), 999999999),
         velox::Timestamp(velox::Timestamp::max().getSeconds(), 999999999)},
        {velox::Timestamp(velox::Timestamp::min().getSeconds(), 0),
         std::nullopt,
         velox::Timestamp(velox::Timestamp::max().getSeconds(), 0)},
        {velox::Timestamp(0, 999999999),
         velox::Timestamp(0, 0),
         std::nullopt}}},
      // Alternating min, max and nulls
      {{{velox::Timestamp::min(),
         std::nullopt,
         velox::Timestamp::max(),
         std::nullopt},
        {std::nullopt,
         velox::Timestamp::max(),
         std::nullopt,
         velox::Timestamp::min()},
        {velox::Timestamp::max(),
         velox::Timestamp::min(),
         std::nullopt,
         std::nullopt}}},
  };

  for (const auto& testCase : testCases) {
    SCOPED_TRACE(testCase.debugString());
    ASSERT_EQ(testCase.columnValues.size(), 3);
    const auto input = makeRowVector(
        {makeNullableFlatVector<velox::Timestamp>(testCase.columnValues[0]),
         makeNullableFlatVector<velox::Timestamp>(testCase.columnValues[1]),
         makeNullableFlatVector<velox::Timestamp>(testCase.columnValues[2])});
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};

    // Test all four sort orders (ASC/DESC x NULLS FIRST/NULLS LAST)
    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, keyColumnsOutOfOrder) {
  // Create a dataset with multiple columns of different types
  // Columns: BIGINT, VARCHAR, DOUBLE, BOOLEAN, INTEGER, SMALLINT
  const auto input = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3, 4, 5}), // c0: BIGINT
      makeFlatVector<std::string>(
          {"apple", "banana", "cherry", "date", "elderberry"}), // c1: VARCHAR
      makeFlatVector<double>({1.5, 2.5, 3.5, 4.5, 5.5}), // c2: DOUBLE
      makeFlatVector<bool>({true, false, true, false, true}), // c3: BOOLEAN
      makeFlatVector<int32_t>({10, 20, 30, 40, 50}), // c4: INTEGER
      makeFlatVector<int16_t>({100, 200, 300, 400, 500}), // c5: SMALLINT
  });

  // Test various key column orderings to ensure KeyEncoder works correctly
  // regardless of column order

  // Original order - all columns
  {
    SCOPED_TRACE("Original order - all columns");
    const std::vector<std::string> keyColumns = {
        "c0", "c1", "c2", "c3", "c4", "c5"};
    encodeTest({input}, keyColumns);
  }

  // Reverse order - all columns
  {
    SCOPED_TRACE("Reverse order - all columns");
    const std::vector<std::string> keyColumns = {
        "c5", "c4", "c3", "c2", "c1", "c0"};
    encodeTest({input}, keyColumns);
  }

  // Random order - subset of columns
  {
    SCOPED_TRACE("Random order - subset of columns");
    const std::vector<std::string> keyColumns = {"c2", "c5", "c0", "c3"};
    encodeTest({input}, keyColumns);
  }

  // Another random order - different subset
  {
    SCOPED_TRACE("Another random order - different subset");
    const std::vector<std::string> keyColumns = {"c4", "c1", "c3"};
    encodeTest({input}, keyColumns);
  }
}

TEST_F(KeyEncoderTest, errorNonExistentColumn) {
  const auto input = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int32_t>({4, 5, 6}),
  });

  const std::vector<std::string> keyColumns = {"c0", "nonexistent"};
  const std::vector<velox::core::SortOrder> sortOrders = {
      velox::core::kAscNullsFirst, velox::core::kAscNullsFirst};

  VELOX_ASSERT_THROW(
      KeyEncoder::create(
          keyColumns, asRowType(input->type()), sortOrders, pool_.get()),
      "Field not found");
}

TEST_F(KeyEncoderTest, errorMismatchedSortOrders) {
  const auto input = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int32_t>({4, 5, 6}),
  });

  const std::vector<std::string> keyColumns = {"c0", "c1"};
  const std::vector<velox::core::SortOrder> sortOrders = {
      velox::core::kAscNullsFirst}; // Only one sort order for two columns

  VELOX_ASSERT_THROW(
      KeyEncoder::create(
          keyColumns, asRowType(input->type()), sortOrders, pool_.get()),
      "Size mismatch");
}

TEST_F(KeyEncoderTest, errorEmptyKeyColumns) {
  const auto input = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int32_t>({4, 5, 6}),
  });

  const std::vector<std::string> keyColumns = {}; // Empty key columns
  const std::vector<velox::core::SortOrder> sortOrders = {};

  VELOX_ASSERT_THROW(
      KeyEncoder::create(
          keyColumns, asRowType(input->type()), sortOrders, pool_.get()),
      "");
}

TEST_F(KeyEncoderTest, sortOrders) {
  const auto input = makeRowVector({
      makeFlatVector<int64_t>({1, 2, 3}),
      makeFlatVector<int32_t>({4, 5, 6}),
      makeFlatVector<std::string>({"a", "b", "c"}),
  });

  // Single column with different sort orders.
  {
    const std::vector<std::string> keyColumns = {"c0"};
    for (const auto& sortOrder :
         {velox::core::kAscNullsFirst,
          velox::core::kAscNullsLast,
          velox::core::kDescNullsFirst,
          velox::core::kDescNullsLast}) {
      const std::vector<velox::core::SortOrder> sortOrders = {sortOrder};
      auto keyEncoder = KeyEncoder::create(
          keyColumns, asRowType(input->type()), sortOrders, pool_.get());
      ASSERT_EQ(keyEncoder->sortOrders().size(), 1);
      EXPECT_EQ(keyEncoder->sortOrders()[0], sortOrder);
    }
  }

  // Multiple columns with different sort orders.
  {
    const std::vector<std::string> keyColumns = {"c0", "c1", "c2"};
    const std::vector<velox::core::SortOrder> sortOrders = {
        velox::core::kAscNullsFirst,
        velox::core::kDescNullsLast,
        velox::core::kAscNullsLast,
    };
    auto keyEncoder = KeyEncoder::create(
        keyColumns, asRowType(input->type()), sortOrders, pool_.get());
    ASSERT_EQ(keyEncoder->sortOrders().size(), 3);
    EXPECT_EQ(keyEncoder->sortOrders()[0], velox::core::kAscNullsFirst);
    EXPECT_EQ(keyEncoder->sortOrders()[1], velox::core::kDescNullsLast);
    EXPECT_EQ(keyEncoder->sortOrders()[2], velox::core::kAscNullsLast);
  }
}

TEST_F(KeyEncoderTest, unsupportedIndexColumnType) {
  // Test ARRAY type
  {
    const auto input = makeRowVector({
        makeArrayVector<int64_t>({{1, 2}, {3, 4}, {5, 6}}),
    });

    const std::vector<std::string> keyColumns = {"c0"};
    const std::vector<velox::core::SortOrder> sortOrders = {
        velox::core::kAscNullsFirst};

    VELOX_ASSERT_THROW(
        KeyEncoder::create(
            keyColumns, asRowType(input->type()), sortOrders, pool_.get()),
        "Unsupported type for index column 'c0': ARRAY<BIGINT>");
  }

  // Test MAP type
  {
    auto input = makeRowVector({
        makeMapVector<int64_t, int64_t>({{{1, 2}, {3, 4}}, {{5, 6}}}),
    });

    const std::vector<std::string> keyColumns = {"c0"};
    const std::vector<velox::core::SortOrder> sortOrders = {
        velox::core::kAscNullsFirst};

    VELOX_ASSERT_THROW(
        KeyEncoder::create(
            keyColumns, asRowType(input->type()), sortOrders, pool_.get()),
        "Unsupported type for index column 'c0': MAP<BIGINT,BIGINT>");
  }

  // Test ROW (struct) type
  {
    auto input = makeRowVector({
        makeRowVector({
            makeFlatVector<int64_t>({1, 2, 3}),
            makeFlatVector<int32_t>({4, 5, 6}),
        }),
    });

    const std::vector<std::string> keyColumns = {"c0"};
    const std::vector<velox::core::SortOrder> sortOrders = {
        velox::core::kAscNullsFirst};

    VELOX_ASSERT_THROW(
        KeyEncoder::create(
            keyColumns, asRowType(input->type()), sortOrders, pool_.get()),
        "Unsupported type for index column 'c0': ROW<c0:BIGINT,c1:INTEGER>");
  }

  // Test UNKNOWN type
  {
    auto input = makeRowVector({
        velox::BaseVector::createNullConstant(velox::UNKNOWN(), 3, pool_.get()),
    });

    const std::vector<std::string> keyColumns = {"c0"};
    const std::vector<velox::core::SortOrder> sortOrders = {
        velox::core::kAscNullsFirst};

    VELOX_ASSERT_THROW(
        KeyEncoder::create(
            keyColumns, asRowType(input->type()), sortOrders, pool_.get()),
        "Unsupported type for index column 'c0': UNKNOWN");
  }

  // Test OPAQUE type
  {
    // Define a simple test class for opaque type
    class TestOpaqueClass {};

    // Create an opaque type vector
    auto opaqueType = velox::OpaqueType::create<TestOpaqueClass>();
    auto opaqueValue = std::make_shared<TestOpaqueClass>();
    auto opaqueVector = velox::BaseVector::createConstant(
        opaqueType, velox::variant::opaque(opaqueValue), 3, pool_.get());

    auto input = makeRowVector({opaqueVector});

    const std::vector<std::string> keyColumns = {"c0"};
    const std::vector<velox::core::SortOrder> sortOrders = {
        velox::core::kAscNullsFirst};

    VELOX_ASSERT_THROW(
        KeyEncoder::create(
            keyColumns, asRowType(input->type()), sortOrders, pool_.get()),
        "Unsupported type for index column 'c0': OPAQUE");
  }

  // Test HUGEINT type
  {
    using HugeintType = velox::TypeTraits<velox::TypeKind::HUGEINT>::NativeType;
    auto input = makeRowVector({
        makeFlatVector<HugeintType>({0, 1}),
    });

    const std::vector<std::string> keyColumns = {"c0"};
    const std::vector<velox::core::SortOrder> sortOrders = {
        velox::core::kAscNullsFirst};

    VELOX_ASSERT_THROW(
        KeyEncoder::create(
            keyColumns, asRowType(input->type()), sortOrders, pool_.get()),
        "Unsupported type for index column 'c0': HUGEINT");
  }
}

TEST_F(KeyEncoderTest, encodeFuzz) {
  // Seed for deterministic testing
  const size_t seed = 123456;

  // Supported scalar types for KeyEncoder (excluding complex types like
  // arrays/maps)
  const std::vector<velox::TypePtr> supportedTypes = {
      velox::BIGINT(),
      velox::INTEGER(),
      velox::SMALLINT(),
      velox::TINYINT(),
      velox::DOUBLE(),
      velox::REAL(),
      velox::BOOLEAN(),
      velox::VARCHAR(),
      velox::DATE(),
      velox::TIMESTAMP()};

  // VectorFuzzer options
  velox::VectorFuzzer::Options options;
  options.vectorSize = 100;
  options.nullRatio = 0.2; // 20% nulls
  options.stringLength = 50;
  options.stringVariableLength = true;
  options.containerLength = 10;
  options.containerVariableLength = true;
  options.allowLazyVector = true;
  options.allowSlice = true;
  options.allowConstantVector = true;
  options.allowDictionaryVector = true;

  // Run fuzzer iterations
  const int numIterations = 50;
  for (int iter = 0; iter < numIterations; ++iter) {
    // Create fuzzer with iteration-specific seed for variation
    velox::VectorFuzzer fuzzer(options, pool_.get(), seed + iter);

    // Generate random number of columns (1 to 5)
    const size_t numColumns = fuzzer.randInRange(1, 5);

    // Generate random row type with supported scalar types
    std::vector<std::string> columnNames;
    std::vector<velox::TypePtr> columnTypes;
    for (size_t i = 0; i < numColumns; ++i) {
      columnNames.push_back(fmt::format("c{}", i));
      // Pick a random supported type
      const size_t typeIndex = fuzzer.randInRange(0, supportedTypes.size() - 1);
      columnTypes.push_back(supportedTypes[typeIndex]);
    }

    // Create row type
    auto rowType = velox::ROW(std::move(columnNames), std::move(columnTypes));

    // Generate multiple input vectors with different encodings (3 to 10
    // vectors per iteration)
    const size_t numInputVectors = fuzzer.randInRange(3, 10);
    std::vector<velox::RowVectorPtr> inputs;
    inputs.reserve(numInputVectors);

    for (size_t i = 0; i < numInputVectors; ++i) {
      // Use fuzzInputRow which generates RowVectors without top-level
      // nulls (i.e., no null rows, only nullable child vectors)
      inputs.push_back(fuzzer.fuzzInputRow(rowType));
    }

    SCOPED_TRACE(
        fmt::format(
            "Iteration {}: {} input vectors, {} columns, types: [{}]",
            iter,
            numInputVectors,
            numColumns,
            folly::join(", ", rowType->names())));

    // Create key column names
    // 80% of the time: randomize the order
    // 20% of the time: follow the same order as input vector columns
    std::vector<std::string> keyColumns;
    keyColumns.reserve(numColumns);
    for (size_t i = 0; i < numColumns; ++i) {
      keyColumns.push_back(fmt::format("c{}", i));
    }

    // Randomize key column order 80% of the time
    if (fuzzer.coinToss(0.8)) {
      // Shuffle the key columns to test if KeyEncoder can handle
      // different orders
      std::mt19937 rng(seed + iter + 1000);
      std::shuffle(keyColumns.begin(), keyColumns.end(), rng);
      SCOPED_TRACE(
          fmt::format(
              "Randomized key column order: [{}]",
              folly::join(", ", keyColumns)));
    } else {
      SCOPED_TRACE("Using original key column order (same as input vector)");
    }

    // Test all input vectors with the same KeyEncoder configuration
    encodeTest(inputs, keyColumns);
  }
}

TEST_F(KeyEncoderTest, encodeIndexBoundsWithBigIntType) {
  // Test Case 1: Both bounds inclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int64_t>({10})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int64_t>({100})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<int64_t>({10})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int64_t>({10})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int64_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int64_t>({10})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 2: Both bounds exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int64_t>({10})}),
            .inclusive = false},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int64_t>({100})}),
            .inclusive = false},
        makeRowVector({makeFlatVector<int64_t>({11})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int64_t>({11})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int64_t>({100})}), // ASC_NULLS_LAST upper
        makeRowVector({makeFlatVector<int64_t>({9})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int64_t>({9})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 3: Lower inclusive, upper exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int64_t>({10})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int64_t>({100})}),
            .inclusive = false},
        makeRowVector({makeFlatVector<int64_t>({10})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int64_t>({10})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int64_t>({100})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int64_t>({10})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 4: Only lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int64_t>({10})}),
            .inclusive = true},
        std::nullopt,
        makeRowVector({makeFlatVector<int64_t>({10})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int64_t>({10})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int64_t>({10})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 5: Only upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int64_t>({100})}),
            .inclusive = true},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({101})}), // ASC_NULLS_FIRST upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int64_t>({101})}), // ASC_NULLS_LAST upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({99})}), // DESC_NULLS_FIRST upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 6: Multi-column, both inclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({10}), makeFlatVector<int64_t>({20})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({100}),
                 makeFlatVector<int64_t>({100})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 7-1: Upper bound at max value (overflow case)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int64_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int64_t>(
                {std::numeric_limits<int64_t>::max()})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<int64_t>({0})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper (overflow)
        makeRowVector({makeFlatVector<int64_t>({0})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper (overflow)
        makeRowVector({makeFlatVector<int64_t>({0})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int64_t>(
            {std::numeric_limits<int64_t>::max() -
             1})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int64_t>({0})}), // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int64_t>(
            {std::numeric_limits<int64_t>::max() -
             1})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 7-2: Upper bound at max value (overflow case)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int64_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int64_t>(
                {std::numeric_limits<int64_t>::min()})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<int64_t>({0})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int64_t>(
            {std::numeric_limits<int64_t>::min() +
             1})}), // ASC_NULLS_FIRST upper (overflow)
        makeRowVector({makeFlatVector<int64_t>({0})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int64_t>(
            {std::numeric_limits<int64_t>::min() +
             1})}), // ASC_NULLS_LAST upper (overflow)
        makeRowVector({makeFlatVector<int64_t>({0})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int64_t>({0})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 8: Only lower bound, exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int64_t>({10})}),
            .inclusive = false},
        std::nullopt,
        makeRowVector({makeFlatVector<int64_t>({11})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int64_t>({11})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector({makeFlatVector<int64_t>({9})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int64_t>({9})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 9: Only upper bound, exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int64_t>({100})}),
            .inclusive = false},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100})}), // ASC_NULLS_FIRST upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int64_t>({100})}), // ASC_NULLS_LAST upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100})}), // DESC_NULLS_FIRST upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 10: Multi-column, both exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({10}), makeFlatVector<int64_t>({20})}),
            .inclusive = false},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({100}),
                 makeFlatVector<int64_t>({100})}),
            .inclusive = false},
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({21})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({100})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({21})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({100})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({19})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({100})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({19})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 11: Multi-column, lower inclusive upper exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({10}), makeFlatVector<int64_t>({20})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({100}),
                 makeFlatVector<int64_t>({100})}),
            .inclusive = false},
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({100})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({100})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({100})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 12: Multi-column, only lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({10}), makeFlatVector<int64_t>({20})}),
            .inclusive = true},
        std::nullopt,
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 13: Multi-column, only upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({100}),
                 makeFlatVector<int64_t>({100})}),
            .inclusive = true},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({101})}), // ASC_NULLS_FIRST upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({101})}), // ASC_NULLS_LAST upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({99})}), // DESC_NULLS_FIRST upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 14-1: Multi-column at max values (overflow)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({1}),
                 makeFlatVector<int64_t>(
                     {std::numeric_limits<int64_t>::max()})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int64_t>({0}),
             makeFlatVector<int64_t>({0})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({2}),
             makeFlatVector<int64_t>(
                 {std::numeric_limits<int64_t>::min()})}), // ASC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeFlatVector<int64_t>({0}),
             makeFlatVector<int64_t>({0})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({2}),
             makeFlatVector<int64_t>(
                 {std::numeric_limits<int64_t>::min()})}), // ASC_NULLS_LAST
                                                           // upper
        makeRowVector(
            {makeFlatVector<int64_t>({0}),
             makeFlatVector<int64_t>({0})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({1}),
             makeFlatVector<int64_t>(
                 {std::numeric_limits<int64_t>::max() -
                  1})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int64_t>({0}),
             makeFlatVector<int64_t>({0})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({1}),
             makeFlatVector<int64_t>(
                 {std::numeric_limits<int64_t>::max() -
                  1})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 14-2: Multi-column at max values
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({1}),
                 makeFlatVector<int64_t>(
                     {std::numeric_limits<int64_t>::min()})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int64_t>({0}),
             makeFlatVector<int64_t>({0})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({1}),
             makeFlatVector<int64_t>(
                 {std::numeric_limits<int64_t>::min() +
                  1})}), // ASC_NULLS_FIRST
                         // upper
        makeRowVector(
            {makeFlatVector<int64_t>({0}),
             makeFlatVector<int64_t>({0})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({1}),
             makeFlatVector<int64_t>(
                 {std::numeric_limits<int64_t>::min() + 1})}), // ASC_NULLS_LAST
                                                               // upper
        makeRowVector(
            {makeFlatVector<int64_t>({0}),
             makeFlatVector<int64_t>({0})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({0}),
             makeFlatVector<int64_t>(
                 {std::numeric_limits<int64_t>::max()})}), // DESC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeFlatVector<int64_t>({0}),
             makeFlatVector<int64_t>({0})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({0}),
             makeFlatVector<int64_t>(
                 {std::numeric_limits<int64_t>::max()})})); // DESC_NULLS_LAST
                                                            // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 14-3: Multi-column at max values (overflow)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({0}), makeFlatVector<int64_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({std::numeric_limits<int64_t>::max()}),
                 makeFlatVector<int64_t>(
                     {std::numeric_limits<int64_t>::max()})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int64_t>({0}),
             makeFlatVector<int64_t>({0})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper (overflow)
        makeRowVector(
            {makeFlatVector<int64_t>({0}),
             makeFlatVector<int64_t>({0})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper (overflow)
        makeRowVector(
            {makeFlatVector<int64_t>({0}),
             makeFlatVector<int64_t>({0})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({std::numeric_limits<int64_t>::max()}),
             makeFlatVector<int64_t>(
                 {std::numeric_limits<int64_t>::max() -
                  1})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int64_t>({0}),
             makeFlatVector<int64_t>({0})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({std::numeric_limits<int64_t>::max()}),
             makeFlatVector<int64_t>(
                 {std::numeric_limits<int64_t>::max() -
                  1})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 15: Multi-column, only lower bound, exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({10}), makeFlatVector<int64_t>({20})}),
            .inclusive = false},
        std::nullopt,
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({21})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({21})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({19})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({19})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 16: Multi-column, only upper bound, exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({100}),
                 makeFlatVector<int64_t>({100})}),
            .inclusive = false},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({100})}), // ASC_NULLS_FIRST upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({100})}), // ASC_NULLS_LAST upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({100})}), // DESC_NULLS_FIRST upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 17: Single column with null in lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<int64_t>({std::nullopt})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int64_t>({100})}),
            .inclusive = true},
        makeRowVector({makeNullableFlatVector<int64_t>(
            {std::nullopt})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeNullableFlatVector<int64_t>(
            {std::nullopt})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int64_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector({makeNullableFlatVector<int64_t>(
            {std::nullopt})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeNullableFlatVector<int64_t>(
            {std::nullopt})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 18: Single column with null in upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int64_t>({10})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<int64_t>({std::nullopt})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<int64_t>({10})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeNullableFlatVector<int64_t>(
            {std::numeric_limits<int64_t>::min()})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int64_t>({10})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeNullableFlatVector<int64_t>(
            {std::numeric_limits<int64_t>::max()})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int64_t>({10})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 19: Multi-column with null in first column of lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<int64_t>({std::nullopt}),
                 makeFlatVector<int64_t>({20})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({100}),
                 makeFlatVector<int64_t>({100})}),
            .inclusive = true},
        makeRowVector(
            {makeNullableFlatVector<int64_t>({std::nullopt}),
             makeFlatVector<int64_t>({20})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeNullableFlatVector<int64_t>({std::nullopt}),
             makeFlatVector<int64_t>({20})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeNullableFlatVector<int64_t>({std::nullopt}),
             makeFlatVector<int64_t>({20})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeNullableFlatVector<int64_t>({std::nullopt}),
             makeFlatVector<int64_t>({20})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 20: Multi-column with null in second column of lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({10}),
                 makeNullableFlatVector<int64_t>({std::nullopt})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({100}),
                 makeFlatVector<int64_t>({100})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeNullableFlatVector<int64_t>(
                 {std::nullopt})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeNullableFlatVector<int64_t>(
                 {std::nullopt})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeNullableFlatVector<int64_t>(
                 {std::nullopt})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeNullableFlatVector<int64_t>(
                 {std::nullopt})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeFlatVector<int64_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 21: Multi-column with null in first column of upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({10}), makeFlatVector<int64_t>({20})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<int64_t>({std::nullopt}),
                 makeFlatVector<int64_t>({100})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeNullableFlatVector<int64_t>({std::nullopt}),
             makeFlatVector<int64_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeNullableFlatVector<int64_t>({std::nullopt}),
             makeFlatVector<int64_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeNullableFlatVector<int64_t>({std::nullopt}),
             makeFlatVector<int64_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeNullableFlatVector<int64_t>({std::nullopt}),
             makeFlatVector<int64_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 22: Multi-column with null in second column of upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({10}), makeFlatVector<int64_t>({20})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({100}),
                 makeNullableFlatVector<int64_t>({std::nullopt})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeNullableFlatVector<int64_t>(
                 {std::numeric_limits<int64_t>::min()})}), // ASC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({101}),
             makeFlatVector<int64_t>(
                 {std::numeric_limits<int64_t>::min()})}), // ASC_NULLS_LAST
                                                           // upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeNullableFlatVector<int64_t>(
                 {std::numeric_limits<int64_t>::max()})}), // DESC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeFlatVector<int64_t>({10}),
             makeFlatVector<int64_t>({20})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({99}),
             makeFlatVector<int64_t>(
                 {std::numeric_limits<int64_t>::max()})})); // DESC_NULLS_LAST
                                                            // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 23: Both bounds with nulls in different columns
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<int64_t>({std::nullopt}),
                 makeFlatVector<int64_t>({20})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int64_t>({100}),
                 makeNullableFlatVector<int64_t>({std::nullopt})}),
            .inclusive = true},
        makeRowVector(
            {makeNullableFlatVector<int64_t>({std::nullopt}),
             makeFlatVector<int64_t>({20})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeNullableFlatVector<int64_t>(
                 {std::numeric_limits<int64_t>::min()})}), // ASC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeNullableFlatVector<int64_t>({std::nullopt}),
             makeFlatVector<int64_t>({20})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({101}),
             makeFlatVector<int64_t>(
                 {std::numeric_limits<int64_t>::min()})}), // ASC_NULLS_LAST
                                                           // upper
        makeRowVector(
            {makeNullableFlatVector<int64_t>({std::nullopt}),
             makeFlatVector<int64_t>({20})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int64_t>({100}),
             makeNullableFlatVector<int64_t>(
                 {std::numeric_limits<int64_t>::max()})}), // DESC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeNullableFlatVector<int64_t>({std::nullopt}),
             makeFlatVector<int64_t>({20})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int64_t>({99}),
             makeFlatVector<int64_t>(
                 {std::numeric_limits<int64_t>::max()})})); // DESC_NULLS_LAST
                                                            // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test lower bound bump failures
  // Test Case 24-1: Single column lower bound bump failure - exclusive bound
  // at max value
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<int64_t>({std::numeric_limits<int64_t>::max()})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsFirst;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }
  // Test Case 24-2: Single column lower bound bump failure - exclusive bound
  // at max value with null.
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound =
            makeRowVector({makeNullableFlatVector<int64_t>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 25-1: Multiple columns lower bound bump failure - all columns
  // at max value
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<int64_t>({std::numeric_limits<int64_t>::max()}),
             makeFlatVector<int64_t>({std::numeric_limits<int64_t>::max()})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }
  // Test Case 25-2: Multiple columns lower bound bump failure - all columns
  // at max value with null.
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<int64_t>({std::nullopt}),
             makeNullableFlatVector<int64_t>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }
}

TEST_F(KeyEncoderTest, encodeIndexBoundsWithIntegerType) {
  // Test Case 1: Both bounds inclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int32_t>({10})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int32_t>({100})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<int32_t>({10})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int32_t>({10})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int32_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int32_t>({10})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 2: Both bounds exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int32_t>({10})}),
            .inclusive = false},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int32_t>({100})}),
            .inclusive = false},
        makeRowVector({makeFlatVector<int32_t>({11})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int32_t>({11})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int32_t>({100})}), // ASC_NULLS_LAST upper
        makeRowVector({makeFlatVector<int32_t>({9})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int32_t>({9})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 3: Lower inclusive, upper exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int32_t>({10})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int32_t>({100})}),
            .inclusive = false},
        makeRowVector({makeFlatVector<int32_t>({10})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int32_t>({10})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int32_t>({100})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int32_t>({10})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 4: Only lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int32_t>({10})}),
            .inclusive = true},
        std::nullopt,
        makeRowVector({makeFlatVector<int32_t>({10})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int32_t>({10})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int32_t>({10})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 5: Only upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int32_t>({100})}),
            .inclusive = true},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({101})}), // ASC_NULLS_FIRST upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int32_t>({101})}), // ASC_NULLS_LAST upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({99})}), // DESC_NULLS_FIRST upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 6: Multi-column, both inclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({10}), makeFlatVector<int32_t>({20})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({100}),
                 makeFlatVector<int32_t>({100})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 7-1: Upper bound at max value (overflow case)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int32_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int32_t>(
                {std::numeric_limits<int32_t>::max()})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<int32_t>({0})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper (overflow)
        makeRowVector({makeFlatVector<int32_t>({0})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper (overflow)
        makeRowVector({makeFlatVector<int32_t>({0})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int32_t>(
            {std::numeric_limits<int32_t>::max() -
             1})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int32_t>({0})}), // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int32_t>(
            {std::numeric_limits<int32_t>::max() -
             1})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 7-2: Upper bound at min value (overflow case)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int32_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int32_t>(
                {std::numeric_limits<int32_t>::min()})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<int32_t>({0})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int32_t>(
            {std::numeric_limits<int32_t>::min() +
             1})}), // ASC_NULLS_FIRST upper (overflow)
        makeRowVector({makeFlatVector<int32_t>({0})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int32_t>(
            {std::numeric_limits<int32_t>::min() +
             1})}), // ASC_NULLS_LAST upper (overflow)
        makeRowVector({makeFlatVector<int32_t>({0})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int32_t>({0})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 8: Only lower bound, exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int32_t>({10})}),
            .inclusive = false},
        std::nullopt,
        makeRowVector({makeFlatVector<int32_t>({11})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int32_t>({11})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector({makeFlatVector<int32_t>({9})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int32_t>({9})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 9: Only upper bound, exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int32_t>({100})}),
            .inclusive = false},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100})}), // ASC_NULLS_FIRST upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int32_t>({100})}), // ASC_NULLS_LAST upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100})}), // DESC_NULLS_FIRST upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 10: Multi-column, both exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({10}), makeFlatVector<int32_t>({20})}),
            .inclusive = false},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({100}),
                 makeFlatVector<int32_t>({100})}),
            .inclusive = false},
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({21})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({100})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({21})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({100})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({19})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({100})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({19})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 11: Multi-column, lower inclusive upper exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({10}), makeFlatVector<int32_t>({20})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({100}),
                 makeFlatVector<int32_t>({100})}),
            .inclusive = false},
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({100})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({100})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({100})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 12: Multi-column, only lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({10}), makeFlatVector<int32_t>({20})}),
            .inclusive = true},
        std::nullopt,
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 13: Multi-column, only upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({100}),
                 makeFlatVector<int32_t>({100})}),
            .inclusive = true},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({101})}), // ASC_NULLS_FIRST upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({101})}), // ASC_NULLS_LAST upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({99})}), // DESC_NULLS_FIRST upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 14-1: Multi-column at max values (overflow)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({0}), makeFlatVector<int32_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({1}),
                 makeFlatVector<int32_t>(
                     {std::numeric_limits<int32_t>::max()})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int32_t>({0}),
             makeFlatVector<int32_t>({0})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({2}),
             makeFlatVector<int32_t>(
                 {std::numeric_limits<int32_t>::min()})}), // ASC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeFlatVector<int32_t>({0}),
             makeFlatVector<int32_t>({0})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({2}),
             makeFlatVector<int32_t>(
                 {std::numeric_limits<int32_t>::min()})}), // ASC_NULLS_LAST
                                                           // upper
        makeRowVector(
            {makeFlatVector<int32_t>({0}),
             makeFlatVector<int32_t>({0})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({1}),
             makeFlatVector<int32_t>(
                 {std::numeric_limits<int32_t>::max() -
                  1})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int32_t>({0}),
             makeFlatVector<int32_t>({0})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({1}),
             makeFlatVector<int32_t>(
                 {std::numeric_limits<int32_t>::max() -
                  1})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 14-2: Multi-column at min values
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({0}), makeFlatVector<int32_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({1}),
                 makeFlatVector<int32_t>(
                     {std::numeric_limits<int32_t>::min()})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int32_t>({0}),
             makeFlatVector<int32_t>({0})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({1}),
             makeFlatVector<int32_t>(
                 {std::numeric_limits<int32_t>::min() +
                  1})}), // ASC_NULLS_FIRST
                         // upper
        makeRowVector(
            {makeFlatVector<int32_t>({0}),
             makeFlatVector<int32_t>({0})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({1}),
             makeFlatVector<int32_t>(
                 {std::numeric_limits<int32_t>::min() + 1})}), // ASC_NULLS_LAST
                                                               // upper
        makeRowVector(
            {makeFlatVector<int32_t>({0}),
             makeFlatVector<int32_t>({0})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({0}),
             makeFlatVector<int32_t>(
                 {std::numeric_limits<int32_t>::max()})}), // DESC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeFlatVector<int32_t>({0}),
             makeFlatVector<int32_t>({0})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({0}),
             makeFlatVector<int32_t>(
                 {std::numeric_limits<int32_t>::max()})})); // DESC_NULLS_LAST
                                                            // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 14-3: Multi-column at max values (overflow)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({0}), makeFlatVector<int32_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({std::numeric_limits<int32_t>::max()}),
                 makeFlatVector<int32_t>(
                     {std::numeric_limits<int32_t>::max()})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int32_t>({0}),
             makeFlatVector<int32_t>({0})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper (overflow)
        makeRowVector(
            {makeFlatVector<int32_t>({0}),
             makeFlatVector<int32_t>({0})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper (overflow)
        makeRowVector(
            {makeFlatVector<int32_t>({0}),
             makeFlatVector<int32_t>({0})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({std::numeric_limits<int32_t>::max()}),
             makeFlatVector<int32_t>(
                 {std::numeric_limits<int32_t>::max() -
                  1})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int32_t>({0}),
             makeFlatVector<int32_t>({0})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({std::numeric_limits<int32_t>::max()}),
             makeFlatVector<int32_t>(
                 {std::numeric_limits<int32_t>::max() -
                  1})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 15: Multi-column, only lower bound, exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({10}), makeFlatVector<int32_t>({20})}),
            .inclusive = false},
        std::nullopt,
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({21})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({21})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({19})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({19})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 16: Multi-column, only upper bound, exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({100}),
                 makeFlatVector<int32_t>({100})}),
            .inclusive = false},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({100})}), // ASC_NULLS_FIRST upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({100})}), // ASC_NULLS_LAST upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({100})}), // DESC_NULLS_FIRST upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 17: Single column with null in lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<int32_t>({std::nullopt})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int32_t>({100})}),
            .inclusive = true},
        makeRowVector({makeNullableFlatVector<int32_t>(
            {std::nullopt})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeNullableFlatVector<int32_t>(
            {std::nullopt})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int32_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector({makeNullableFlatVector<int32_t>(
            {std::nullopt})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeNullableFlatVector<int32_t>(
            {std::nullopt})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 18: Single column with null in upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int32_t>({10})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<int32_t>({std::nullopt})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<int32_t>({10})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeNullableFlatVector<int32_t>(
            {std::numeric_limits<int32_t>::min()})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int32_t>({10})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeNullableFlatVector<int32_t>(
            {std::numeric_limits<int32_t>::max()})}), // DESC_NULLS_FIRST
                                                      // upper
        makeRowVector({makeFlatVector<int32_t>({10})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 19: Multi-column with null in first column of lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<int32_t>({std::nullopt}),
                 makeFlatVector<int32_t>({20})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({100}),
                 makeFlatVector<int32_t>({100})}),
            .inclusive = true},
        makeRowVector(
            {makeNullableFlatVector<int32_t>({std::nullopt}),
             makeFlatVector<int32_t>({20})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeNullableFlatVector<int32_t>({std::nullopt}),
             makeFlatVector<int32_t>({20})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeNullableFlatVector<int32_t>({std::nullopt}),
             makeFlatVector<int32_t>({20})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeNullableFlatVector<int32_t>({std::nullopt}),
             makeFlatVector<int32_t>({20})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 20: Multi-column with null in second column of lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({10}),
                 makeNullableFlatVector<int32_t>({std::nullopt})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({100}),
                 makeFlatVector<int32_t>({100})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeNullableFlatVector<int32_t>(
                 {std::nullopt})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeNullableFlatVector<int32_t>(
                 {std::nullopt})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeNullableFlatVector<int32_t>(
                 {std::nullopt})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeNullableFlatVector<int32_t>(
                 {std::nullopt})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeFlatVector<int32_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 21: Multi-column with null in first column of upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({10}), makeFlatVector<int32_t>({20})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<int32_t>({std::nullopt}),
                 makeFlatVector<int32_t>({100})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeNullableFlatVector<int32_t>({std::nullopt}),
             makeFlatVector<int32_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeNullableFlatVector<int32_t>({std::nullopt}),
             makeFlatVector<int32_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeNullableFlatVector<int32_t>({std::nullopt}),
             makeFlatVector<int32_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeNullableFlatVector<int32_t>({std::nullopt}),
             makeFlatVector<int32_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 22: Multi-column with null in second column of upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({10}), makeFlatVector<int32_t>({20})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({100}),
                 makeNullableFlatVector<int32_t>({std::nullopt})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeNullableFlatVector<int32_t>(
                 {std::numeric_limits<int32_t>::min()})}), // ASC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({101}),
             makeFlatVector<int32_t>(
                 {std::numeric_limits<int32_t>::min()})}), // ASC_NULLS_LAST
                                                           // upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeNullableFlatVector<int32_t>(
                 {std::numeric_limits<int32_t>::max()})}), // DESC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeFlatVector<int32_t>({10}),
             makeFlatVector<int32_t>({20})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({99}),
             makeFlatVector<int32_t>(
                 {std::numeric_limits<int32_t>::max()})})); // DESC_NULLS_LAST
                                                            // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 23: Both bounds with nulls in different columns
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<int32_t>({std::nullopt}),
                 makeFlatVector<int32_t>({20})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int32_t>({100}),
                 makeNullableFlatVector<int32_t>({std::nullopt})}),
            .inclusive = true},
        makeRowVector(
            {makeNullableFlatVector<int32_t>({std::nullopt}),
             makeFlatVector<int32_t>({20})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeNullableFlatVector<int32_t>(
                 {std::numeric_limits<int32_t>::min()})}), // ASC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeNullableFlatVector<int32_t>({std::nullopt}),
             makeFlatVector<int32_t>({20})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({101}),
             makeFlatVector<int32_t>(
                 {std::numeric_limits<int32_t>::min()})}), // ASC_NULLS_LAST
                                                           // upper
        makeRowVector(
            {makeNullableFlatVector<int32_t>({std::nullopt}),
             makeFlatVector<int32_t>({20})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int32_t>({100}),
             makeNullableFlatVector<int32_t>(
                 {std::numeric_limits<int32_t>::max()})}), // DESC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeNullableFlatVector<int32_t>({std::nullopt}),
             makeFlatVector<int32_t>({20})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int32_t>({99}),
             makeFlatVector<int32_t>(
                 {std::numeric_limits<int32_t>::max()})})); // DESC_NULLS_LAST
                                                            // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test lower bound bump failures
  // Test Case 24-1: Single column lower bound bump failure - exclusive bound
  // at max value
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<int32_t>({std::numeric_limits<int32_t>::max()})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsFirst;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }
  // Test Case 24-2: Single column lower bound bump failure - exclusive bound
  // at max value with null.
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound =
            makeRowVector({makeNullableFlatVector<int32_t>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 25-1: Multiple columns lower bound bump failure - all columns
  // at max value
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<int32_t>({std::numeric_limits<int32_t>::max()}),
             makeFlatVector<int32_t>({std::numeric_limits<int32_t>::max()})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }
  // Test Case 25-2: Multiple columns lower bound bump failure - all columns
  // at max value with null.
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<int32_t>({std::nullopt}),
             makeNullableFlatVector<int32_t>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }
}

TEST_F(KeyEncoderTest, encodeIndexBoundsWithSmallIntType) {
  // Test Case 1: Both bounds inclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int16_t>({10})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int16_t>({100})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<int16_t>({10})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int16_t>({10})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int16_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int16_t>({10})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 2: Both bounds exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int16_t>({10})}),
            .inclusive = false},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int16_t>({100})}),
            .inclusive = false},
        makeRowVector({makeFlatVector<int16_t>({11})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int16_t>({11})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int16_t>({100})}), // ASC_NULLS_LAST upper
        makeRowVector({makeFlatVector<int16_t>({9})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int16_t>({9})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 3: Lower inclusive, upper exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int16_t>({10})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int16_t>({100})}),
            .inclusive = false},
        makeRowVector({makeFlatVector<int16_t>({10})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int16_t>({10})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int16_t>({100})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int16_t>({10})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 4: Only lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int16_t>({10})}),
            .inclusive = true},
        std::nullopt,
        makeRowVector({makeFlatVector<int16_t>({10})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int16_t>({10})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int16_t>({10})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 5: Only upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int16_t>({100})}),
            .inclusive = true},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({101})}), // ASC_NULLS_FIRST upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int16_t>({101})}), // ASC_NULLS_LAST upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({99})}), // DESC_NULLS_FIRST upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 6: Multi-column, both inclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({10}), makeFlatVector<int16_t>({20})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({100}),
                 makeFlatVector<int16_t>({100})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 7-1: Upper bound at max value (overflow case)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int16_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int16_t>(
                {std::numeric_limits<int16_t>::max()})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<int16_t>({0})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper (overflow)
        makeRowVector({makeFlatVector<int16_t>({0})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper (overflow)
        makeRowVector({makeFlatVector<int16_t>({0})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int16_t>(
            {std::numeric_limits<int16_t>::max() -
             1})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int16_t>({0})}), // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int16_t>(
            {std::numeric_limits<int16_t>::max() -
             1})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 7-2: Upper bound at min value (overflow case)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int16_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int16_t>(
                {std::numeric_limits<int16_t>::min()})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<int16_t>({0})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int16_t>(
            {std::numeric_limits<int16_t>::min() +
             1})}), // ASC_NULLS_FIRST upper (overflow)
        makeRowVector({makeFlatVector<int16_t>({0})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int16_t>(
            {std::numeric_limits<int16_t>::min() +
             1})}), // ASC_NULLS_LAST upper (overflow)
        makeRowVector({makeFlatVector<int16_t>({0})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int16_t>({0})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 8: Only lower bound, exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int16_t>({10})}),
            .inclusive = false},
        std::nullopt,
        makeRowVector({makeFlatVector<int16_t>({11})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int16_t>({11})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector({makeFlatVector<int16_t>({9})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int16_t>({9})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 9: Only upper bound, exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int16_t>({100})}),
            .inclusive = false},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100})}), // ASC_NULLS_FIRST upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int16_t>({100})}), // ASC_NULLS_LAST upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100})}), // DESC_NULLS_FIRST upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 10: Multi-column, both exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({10}), makeFlatVector<int16_t>({20})}),
            .inclusive = false},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({100}),
                 makeFlatVector<int16_t>({100})}),
            .inclusive = false},
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({21})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({100})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({21})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({100})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({19})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({100})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({19})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 11: Multi-column, lower inclusive upper exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({10}), makeFlatVector<int16_t>({20})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({100}),
                 makeFlatVector<int16_t>({100})}),
            .inclusive = false},
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({100})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({100})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({100})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 12: Multi-column, only lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({10}), makeFlatVector<int16_t>({20})}),
            .inclusive = true},
        std::nullopt,
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 13: Multi-column, only upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({100}),
                 makeFlatVector<int16_t>({100})}),
            .inclusive = true},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({101})}), // ASC_NULLS_FIRST upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({101})}), // ASC_NULLS_LAST upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({99})}), // DESC_NULLS_FIRST upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 14-1: Multi-column at max values (overflow)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({0}), makeFlatVector<int16_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({1}),
                 makeFlatVector<int16_t>(
                     {std::numeric_limits<int16_t>::max()})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int16_t>({0}),
             makeFlatVector<int16_t>({0})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({2}),
             makeFlatVector<int16_t>(
                 {std::numeric_limits<int16_t>::min()})}), // ASC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeFlatVector<int16_t>({0}),
             makeFlatVector<int16_t>({0})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({2}),
             makeFlatVector<int16_t>(
                 {std::numeric_limits<int16_t>::min()})}), // ASC_NULLS_LAST
                                                           // upper
        makeRowVector(
            {makeFlatVector<int16_t>({0}),
             makeFlatVector<int16_t>({0})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({1}),
             makeFlatVector<int16_t>(
                 {std::numeric_limits<int16_t>::max() -
                  1})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int16_t>({0}),
             makeFlatVector<int16_t>({0})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({1}),
             makeFlatVector<int16_t>(
                 {std::numeric_limits<int16_t>::max() -
                  1})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 14-2: Multi-column at min values
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({0}), makeFlatVector<int16_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({1}),
                 makeFlatVector<int16_t>(
                     {std::numeric_limits<int16_t>::min()})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int16_t>({0}),
             makeFlatVector<int16_t>({0})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({1}),
             makeFlatVector<int16_t>(
                 {std::numeric_limits<int16_t>::min() +
                  1})}), // ASC_NULLS_FIRST
                         // upper
        makeRowVector(
            {makeFlatVector<int16_t>({0}),
             makeFlatVector<int16_t>({0})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({1}),
             makeFlatVector<int16_t>(
                 {std::numeric_limits<int16_t>::min() + 1})}), // ASC_NULLS_LAST
                                                               // upper
        makeRowVector(
            {makeFlatVector<int16_t>({0}),
             makeFlatVector<int16_t>({0})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({0}),
             makeFlatVector<int16_t>(
                 {std::numeric_limits<int16_t>::max()})}), // DESC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeFlatVector<int16_t>({0}),
             makeFlatVector<int16_t>({0})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({0}),
             makeFlatVector<int16_t>(
                 {std::numeric_limits<int16_t>::max()})})); // DESC_NULLS_LAST
                                                            // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 14-3: Multi-column at max values (overflow)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({0}), makeFlatVector<int16_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({std::numeric_limits<int16_t>::max()}),
                 makeFlatVector<int16_t>(
                     {std::numeric_limits<int16_t>::max()})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int16_t>({0}),
             makeFlatVector<int16_t>({0})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper (overflow)
        makeRowVector(
            {makeFlatVector<int16_t>({0}),
             makeFlatVector<int16_t>({0})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper (overflow)
        makeRowVector(
            {makeFlatVector<int16_t>({0}),
             makeFlatVector<int16_t>({0})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({std::numeric_limits<int16_t>::max()}),
             makeFlatVector<int16_t>(
                 {std::numeric_limits<int16_t>::max() -
                  1})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int16_t>({0}),
             makeFlatVector<int16_t>({0})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({std::numeric_limits<int16_t>::max()}),
             makeFlatVector<int16_t>(
                 {std::numeric_limits<int16_t>::max() -
                  1})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 15: Multi-column, only lower bound, exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({10}), makeFlatVector<int16_t>({20})}),
            .inclusive = false},
        std::nullopt,
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({21})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({21})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({19})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({19})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 16: Multi-column, only upper bound, exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({100}),
                 makeFlatVector<int16_t>({100})}),
            .inclusive = false},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({100})}), // ASC_NULLS_FIRST upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({100})}), // ASC_NULLS_LAST upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({100})}), // DESC_NULLS_FIRST upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 17: Single column with null in lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<int16_t>({std::nullopt})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int16_t>({100})}),
            .inclusive = true},
        makeRowVector({makeNullableFlatVector<int16_t>(
            {std::nullopt})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeNullableFlatVector<int16_t>(
            {std::nullopt})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int16_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector({makeNullableFlatVector<int16_t>(
            {std::nullopt})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeNullableFlatVector<int16_t>(
            {std::nullopt})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 18: Single column with null in upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int16_t>({10})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<int16_t>({std::nullopt})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<int16_t>({10})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeNullableFlatVector<int16_t>(
            {std::numeric_limits<int16_t>::min()})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int16_t>({10})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeNullableFlatVector<int16_t>(
            {std::numeric_limits<int16_t>::max()})}), // DESC_NULLS_FIRST
                                                      // upper
        makeRowVector({makeFlatVector<int16_t>({10})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 19: Multi-column with null in first column of lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<int16_t>({std::nullopt}),
                 makeFlatVector<int16_t>({20})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({100}),
                 makeFlatVector<int16_t>({100})}),
            .inclusive = true},
        makeRowVector(
            {makeNullableFlatVector<int16_t>({std::nullopt}),
             makeFlatVector<int16_t>({20})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeNullableFlatVector<int16_t>({std::nullopt}),
             makeFlatVector<int16_t>({20})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeNullableFlatVector<int16_t>({std::nullopt}),
             makeFlatVector<int16_t>({20})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeNullableFlatVector<int16_t>({std::nullopt}),
             makeFlatVector<int16_t>({20})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 20: Multi-column with null in second column of lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({10}),
                 makeNullableFlatVector<int16_t>({std::nullopt})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({100}),
                 makeFlatVector<int16_t>({100})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeNullableFlatVector<int16_t>(
                 {std::nullopt})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeNullableFlatVector<int16_t>(
                 {std::nullopt})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeNullableFlatVector<int16_t>(
                 {std::nullopt})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeNullableFlatVector<int16_t>(
                 {std::nullopt})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeFlatVector<int16_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 21: Multi-column with null in first column of upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({10}), makeFlatVector<int16_t>({20})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<int16_t>({std::nullopt}),
                 makeFlatVector<int16_t>({100})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeNullableFlatVector<int16_t>({std::nullopt}),
             makeFlatVector<int16_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeNullableFlatVector<int16_t>({std::nullopt}),
             makeFlatVector<int16_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeNullableFlatVector<int16_t>({std::nullopt}),
             makeFlatVector<int16_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeNullableFlatVector<int16_t>({std::nullopt}),
             makeFlatVector<int16_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 22: Multi-column with null in second column of upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({10}), makeFlatVector<int16_t>({20})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({100}),
                 makeNullableFlatVector<int16_t>({std::nullopt})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeNullableFlatVector<int16_t>(
                 {std::numeric_limits<int16_t>::min()})}), // ASC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({101}),
             makeFlatVector<int16_t>(
                 {std::numeric_limits<int16_t>::min()})}), // ASC_NULLS_LAST
                                                           // upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeNullableFlatVector<int16_t>(
                 {std::numeric_limits<int16_t>::max()})}), // DESC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeFlatVector<int16_t>({10}),
             makeFlatVector<int16_t>({20})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({99}),
             makeFlatVector<int16_t>(
                 {std::numeric_limits<int16_t>::max()})})); // DESC_NULLS_LAST
                                                            // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 23: Both bounds with nulls in different columns
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<int16_t>({std::nullopt}),
                 makeFlatVector<int16_t>({20})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int16_t>({100}),
                 makeNullableFlatVector<int16_t>({std::nullopt})}),
            .inclusive = true},
        makeRowVector(
            {makeNullableFlatVector<int16_t>({std::nullopt}),
             makeFlatVector<int16_t>({20})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeNullableFlatVector<int16_t>(
                 {std::numeric_limits<int16_t>::min()})}), // ASC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeNullableFlatVector<int16_t>({std::nullopt}),
             makeFlatVector<int16_t>({20})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({101}),
             makeFlatVector<int16_t>(
                 {std::numeric_limits<int16_t>::min()})}), // ASC_NULLS_LAST
                                                           // upper
        makeRowVector(
            {makeNullableFlatVector<int16_t>({std::nullopt}),
             makeFlatVector<int16_t>({20})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int16_t>({100}),
             makeNullableFlatVector<int16_t>(
                 {std::numeric_limits<int16_t>::max()})}), // DESC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeNullableFlatVector<int16_t>({std::nullopt}),
             makeFlatVector<int16_t>({20})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int16_t>({99}),
             makeFlatVector<int16_t>(
                 {std::numeric_limits<int16_t>::max()})})); // DESC_NULLS_LAST
                                                            // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test lower bound bump failures
  // Test Case 24-1: Single column lower bound bump failure - exclusive
  // bound at max value
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<int16_t>({std::numeric_limits<int16_t>::max()})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsFirst;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }
  // Test Case 24-2: Single column lower bound bump failure - exclusive
  // bound at max value with null.
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound =
            makeRowVector({makeNullableFlatVector<int16_t>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 25-1: Multiple columns lower bound bump failure - all
  // columns at max value
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<int16_t>({std::numeric_limits<int16_t>::max()}),
             makeFlatVector<int16_t>({std::numeric_limits<int16_t>::max()})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }
  // Test Case 25-2: Multiple columns lower bound bump failure - all
  // columns at max value with null.
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<int16_t>({std::nullopt}),
             makeNullableFlatVector<int16_t>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }
}

TEST_F(KeyEncoderTest, encodeIndexBoundsWithTinyIntType) {
  // Test Case 1: Both bounds inclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int8_t>({5})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int8_t>({50})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<int8_t>({5})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int8_t>({51})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int8_t>({5})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int8_t>({51})}), // ASC_NULLS_LAST upper
        makeRowVector({makeFlatVector<int8_t>({5})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int8_t>({49})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int8_t>({5})}), // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int8_t>({49})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 2: Both bounds exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int8_t>({5})}),
            .inclusive = false},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int8_t>({50})}),
            .inclusive = false},
        makeRowVector({makeFlatVector<int8_t>({6})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int8_t>({50})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int8_t>({6})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int8_t>({50})}), // ASC_NULLS_LAST upper
        makeRowVector({makeFlatVector<int8_t>({4})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int8_t>({50})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int8_t>({4})}), // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int8_t>({50})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 3: Lower inclusive, upper exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int8_t>({5})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int8_t>({50})}),
            .inclusive = false},
        makeRowVector({makeFlatVector<int8_t>({5})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int8_t>({50})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int8_t>({5})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int8_t>({50})}), // ASC_NULLS_LAST upper
        makeRowVector({makeFlatVector<int8_t>({5})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int8_t>({50})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int8_t>({5})}), // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int8_t>({50})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 4: Only lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int8_t>({5})}),
            .inclusive = true},
        std::nullopt,
        makeRowVector({makeFlatVector<int8_t>({5})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int8_t>({5})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector({makeFlatVector<int8_t>({5})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int8_t>({5})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 5: Only upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int8_t>({50})}),
            .inclusive = true},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int8_t>({51})}), // ASC_NULLS_FIRST upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int8_t>({51})}), // ASC_NULLS_LAST upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int8_t>({49})}), // DESC_NULLS_FIRST upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int8_t>({49})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 6: Multi-column, both inclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({5}), makeFlatVector<int8_t>({10})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({50}), makeFlatVector<int8_t>({100})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 7-1: Upper bound at max value (overflow case)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int8_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({std::numeric_limits<int8_t>::max()})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<int8_t>({0})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper (overflow)
        makeRowVector({makeFlatVector<int8_t>({0})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper (overflow)
        makeRowVector({makeFlatVector<int8_t>({0})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int8_t>(
            {std::numeric_limits<int8_t>::max() -
             1})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int8_t>({0})}), // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int8_t>(
            {std::numeric_limits<int8_t>::max() -
             1})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 7-2: Upper bound at min value (overflow case)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int8_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({std::numeric_limits<int8_t>::min()})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<int8_t>({0})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int8_t>(
            {std::numeric_limits<int8_t>::min() +
             1})}), // ASC_NULLS_FIRST upper (overflow)
        makeRowVector({makeFlatVector<int8_t>({0})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int8_t>(
            {std::numeric_limits<int8_t>::min() +
             1})}), // ASC_NULLS_LAST upper (overflow)
        makeRowVector({makeFlatVector<int8_t>({0})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int8_t>({0})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 8: Only lower bound, exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int8_t>({5})}),
            .inclusive = false},
        std::nullopt,
        makeRowVector({makeFlatVector<int8_t>({6})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int8_t>({6})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector({makeFlatVector<int8_t>({4})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<int8_t>({4})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 9: Only upper bound, exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int8_t>({50})}),
            .inclusive = false},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int8_t>({50})}), // ASC_NULLS_FIRST upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int8_t>({50})}), // ASC_NULLS_LAST upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int8_t>({50})}), // DESC_NULLS_FIRST upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int8_t>({50})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 10: Multi-column, both exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({5}), makeFlatVector<int8_t>({10})}),
            .inclusive = false},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({50}), makeFlatVector<int8_t>({100})}),
            .inclusive = false},
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({11})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({100})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({11})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({100})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({9})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({100})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({9})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 11: Multi-column, lower inclusive upper exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({5}), makeFlatVector<int8_t>({10})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({50}), makeFlatVector<int8_t>({100})}),
            .inclusive = false},
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({100})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({100})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({100})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 12: Multi-column, only lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({5}), makeFlatVector<int8_t>({10})}),
            .inclusive = true},
        std::nullopt,
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 13: Multi-column, only upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({50}), makeFlatVector<int8_t>({100})}),
            .inclusive = true},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({101})}), // ASC_NULLS_FIRST upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({101})}), // ASC_NULLS_LAST upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({99})}), // DESC_NULLS_FIRST upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 14-1: Multi-column at max values (overflow)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({0}), makeFlatVector<int8_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({1}),
                 makeFlatVector<int8_t>({std::numeric_limits<int8_t>::max()})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int8_t>({0}),
             makeFlatVector<int8_t>({0})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({2}),
             makeFlatVector<int8_t>(
                 {std::numeric_limits<int8_t>::min()})}), // ASC_NULLS_FIRST
                                                          // upper
        makeRowVector(
            {makeFlatVector<int8_t>({0}),
             makeFlatVector<int8_t>({0})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({2}),
             makeFlatVector<int8_t>(
                 {std::numeric_limits<int8_t>::min()})}), // ASC_NULLS_LAST
                                                          // upper
        makeRowVector(
            {makeFlatVector<int8_t>({0}),
             makeFlatVector<int8_t>({0})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({1}),
             makeFlatVector<int8_t>(
                 {std::numeric_limits<int8_t>::max() -
                  1})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int8_t>({0}),
             makeFlatVector<int8_t>({0})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({1}),
             makeFlatVector<int8_t>(
                 {std::numeric_limits<int8_t>::max() -
                  1})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 14-2: Multi-column at min values
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({0}), makeFlatVector<int8_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({1}),
                 makeFlatVector<int8_t>({std::numeric_limits<int8_t>::min()})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int8_t>({0}),
             makeFlatVector<int8_t>({0})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({1}),
             makeFlatVector<int8_t>(
                 {std::numeric_limits<int8_t>::min() + 1})}), // ASC_NULLS_FIRST
                                                              // upper
        makeRowVector(
            {makeFlatVector<int8_t>({0}),
             makeFlatVector<int8_t>({0})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({1}),
             makeFlatVector<int8_t>(
                 {std::numeric_limits<int8_t>::min() + 1})}), // ASC_NULLS_LAST
                                                              // upper
        makeRowVector(
            {makeFlatVector<int8_t>({0}),
             makeFlatVector<int8_t>({0})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({0}),
             makeFlatVector<int8_t>(
                 {std::numeric_limits<int8_t>::max()})}), // DESC_NULLS_FIRST
                                                          // upper
        makeRowVector(
            {makeFlatVector<int8_t>({0}),
             makeFlatVector<int8_t>({0})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({0}),
             makeFlatVector<int8_t>(
                 {std::numeric_limits<int8_t>::max()})})); // DESC_NULLS_LAST
                                                           // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 14-3: Multi-column at max values (overflow)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({0}), makeFlatVector<int8_t>({0})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({std::numeric_limits<int8_t>::max()}),
                 makeFlatVector<int8_t>({std::numeric_limits<int8_t>::max()})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int8_t>({0}),
             makeFlatVector<int8_t>({0})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper (overflow)
        makeRowVector(
            {makeFlatVector<int8_t>({0}),
             makeFlatVector<int8_t>({0})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper (overflow)
        makeRowVector(
            {makeFlatVector<int8_t>({0}),
             makeFlatVector<int8_t>({0})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({std::numeric_limits<int8_t>::max()}),
             makeFlatVector<int8_t>(
                 {std::numeric_limits<int8_t>::max() -
                  1})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int8_t>({0}),
             makeFlatVector<int8_t>({0})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({std::numeric_limits<int8_t>::max()}),
             makeFlatVector<int8_t>(
                 {std::numeric_limits<int8_t>::max() -
                  1})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 15: Multi-column, only lower bound, exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({5}), makeFlatVector<int8_t>({10})}),
            .inclusive = false},
        std::nullopt,
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({11})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({11})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({9})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({9})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 16: Multi-column, only upper bound, exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({50}), makeFlatVector<int8_t>({100})}),
            .inclusive = false},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({100})}), // ASC_NULLS_FIRST upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({100})}), // ASC_NULLS_LAST upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({100})}), // DESC_NULLS_FIRST upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({100})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 17: Single column with null in lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound =
                makeRowVector({makeNullableFlatVector<int8_t>({std::nullopt})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int8_t>({50})}),
            .inclusive = true},
        makeRowVector({makeNullableFlatVector<int8_t>(
            {std::nullopt})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int8_t>({51})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeNullableFlatVector<int8_t>(
            {std::nullopt})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int8_t>({51})}), // ASC_NULLS_LAST upper
        makeRowVector({makeNullableFlatVector<int8_t>(
            {std::nullopt})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<int8_t>({49})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeNullableFlatVector<int8_t>(
            {std::nullopt})}), // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<int8_t>({49})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 18: Single column with null in upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<int8_t>({5})}),
            .inclusive = true},
        IndexBound{
            .bound =
                makeRowVector({makeNullableFlatVector<int8_t>({std::nullopt})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<int8_t>({5})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeNullableFlatVector<int8_t>(
            {std::numeric_limits<int8_t>::min()})}), // ASC_NULLS_FIRST
                                                     // upper
        makeRowVector({makeFlatVector<int8_t>({5})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector({makeFlatVector<int8_t>({5})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeNullableFlatVector<int8_t>(
            {std::numeric_limits<int8_t>::max()})}), // DESC_NULLS_FIRST
                                                     // upper
        makeRowVector({makeFlatVector<int8_t>({5})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 19: Multi-column with null in first column of lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<int8_t>({std::nullopt}),
                 makeFlatVector<int8_t>({10})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({50}), makeFlatVector<int8_t>({100})}),
            .inclusive = true},
        makeRowVector(
            {makeNullableFlatVector<int8_t>({std::nullopt}),
             makeFlatVector<int8_t>({10})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeNullableFlatVector<int8_t>({std::nullopt}),
             makeFlatVector<int8_t>({10})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeNullableFlatVector<int8_t>({std::nullopt}),
             makeFlatVector<int8_t>({10})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeNullableFlatVector<int8_t>({std::nullopt}),
             makeFlatVector<int8_t>({10})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 20: Multi-column with null in second column of lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({5}),
                 makeNullableFlatVector<int8_t>({std::nullopt})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({50}), makeFlatVector<int8_t>({100})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeNullableFlatVector<int8_t>(
                 {std::nullopt})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeNullableFlatVector<int8_t>(
                 {std::nullopt})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeNullableFlatVector<int8_t>(
                 {std::nullopt})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeNullableFlatVector<int8_t>(
                 {std::nullopt})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeFlatVector<int8_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 21: Multi-column with null in first column of upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({5}), makeFlatVector<int8_t>({10})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<int8_t>({std::nullopt}),
                 makeFlatVector<int8_t>({100})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeNullableFlatVector<int8_t>({std::nullopt}),
             makeFlatVector<int8_t>({101})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeNullableFlatVector<int8_t>({std::nullopt}),
             makeFlatVector<int8_t>({101})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeNullableFlatVector<int8_t>({std::nullopt}),
             makeFlatVector<int8_t>({99})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeNullableFlatVector<int8_t>({std::nullopt}),
             makeFlatVector<int8_t>({99})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 22: Multi-column with null in second column of upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({5}), makeFlatVector<int8_t>({10})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({50}),
                 makeNullableFlatVector<int8_t>({std::nullopt})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeNullableFlatVector<int8_t>(
                 {std::numeric_limits<int8_t>::min()})}), // ASC_NULLS_FIRST
                                                          // upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({51}),
             makeFlatVector<int8_t>(
                 {std::numeric_limits<int8_t>::min()})}), // ASC_NULLS_LAST
                                                          // upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeNullableFlatVector<int8_t>(
                 {std::numeric_limits<int8_t>::max()})}), // DESC_NULLS_FIRST
                                                          // upper
        makeRowVector(
            {makeFlatVector<int8_t>({5}),
             makeFlatVector<int8_t>({10})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({49}),
             makeFlatVector<int8_t>(
                 {std::numeric_limits<int8_t>::max()})})); // DESC_NULLS_LAST
                                                           // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 23: Both bounds with nulls in different columns
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<int8_t>({std::nullopt}),
                 makeFlatVector<int8_t>({10})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<int8_t>({50}),
                 makeNullableFlatVector<int8_t>({std::nullopt})}),
            .inclusive = true},
        makeRowVector(
            {makeNullableFlatVector<int8_t>({std::nullopt}),
             makeFlatVector<int8_t>({10})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeNullableFlatVector<int8_t>(
                 {std::numeric_limits<int8_t>::min()})}), // ASC_NULLS_FIRST
                                                          // upper
        makeRowVector(
            {makeNullableFlatVector<int8_t>({std::nullopt}),
             makeFlatVector<int8_t>({10})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({51}),
             makeFlatVector<int8_t>(
                 {std::numeric_limits<int8_t>::min()})}), // ASC_NULLS_LAST
                                                          // upper
        makeRowVector(
            {makeNullableFlatVector<int8_t>({std::nullopt}),
             makeFlatVector<int8_t>({10})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<int8_t>({50}),
             makeNullableFlatVector<int8_t>(
                 {std::numeric_limits<int8_t>::max()})}), // DESC_NULLS_FIRST
                                                          // upper
        makeRowVector(
            {makeNullableFlatVector<int8_t>({std::nullopt}),
             makeFlatVector<int8_t>({10})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<int8_t>({49}),
             makeFlatVector<int8_t>(
                 {std::numeric_limits<int8_t>::max()})})); // DESC_NULLS_LAST
                                                           // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test lower bound bump failures
  // Test Case 24-1: Single column lower bound bump failure - exclusive
  // bound at max value
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<int8_t>({std::numeric_limits<int8_t>::max()})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsFirst;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }
  // Test Case 24-2: Single column lower bound bump failure - exclusive
  // bound at max value with null.
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound =
            makeRowVector({makeNullableFlatVector<int8_t>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 25-1: Multiple columns lower bound bump failure - all
  // columns at max value
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<int8_t>({std::numeric_limits<int8_t>::max()}),
             makeFlatVector<int8_t>({std::numeric_limits<int8_t>::max()})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }
  // Test Case 25-2: Multiple columns lower bound bump failure - all
  // columns at max value with null.
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<int8_t>({std::nullopt}),
             makeNullableFlatVector<int8_t>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }
}

TEST_F(KeyEncoderTest, encodeIndexBoundsWithBoolType) {
  // Test Case 1: Both bounds inclusive (false to true)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<bool>({false})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<bool>({true})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<bool>({false})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper (overflow - can't increment
                      // true)
        makeRowVector({makeFlatVector<bool>({false})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper (overflow)
        makeRowVector(
            {makeFlatVector<bool>({false})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<bool>({false})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<bool>({false})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<bool>({false})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 2: Upper bound exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<bool>({false})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<bool>({true})}),
            .inclusive = false},
        makeRowVector({makeFlatVector<bool>({false})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<bool>({true})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<bool>({false})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<bool>({true})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<bool>({false})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<bool>({true})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<bool>({false})}), // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<bool>({true})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 3: Only lower bound (false)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<bool>({false})}),
            .inclusive = true},
        std::nullopt,
        makeRowVector({makeFlatVector<bool>({false})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<bool>({false})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<bool>({false})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<bool>({false})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 4: Only upper bound (true)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector({makeFlatVector<bool>({true})}),
            .inclusive = true},
        std::nullopt, // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper (overflow)
        std::nullopt, // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper (overflow)
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<bool>({false})}), // DESC_NULLS_FIRST upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<bool>({false})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 5: Single column with null in lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound =
                makeRowVector({makeNullableFlatVector<bool>({std::nullopt})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<bool>({true})}),
            .inclusive = true},
        makeRowVector({makeNullableFlatVector<bool>(
            {std::nullopt})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper (overflow)
        makeRowVector({makeNullableFlatVector<bool>(
            {std::nullopt})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper (overflow)
        makeRowVector({makeNullableFlatVector<bool>(
            {std::nullopt})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<bool>({false})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeNullableFlatVector<bool>(
            {std::nullopt})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<bool>({false})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 6: Single column with null in upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<bool>({false})}),
            .inclusive = true},
        IndexBound{
            .bound =
                makeRowVector({makeNullableFlatVector<bool>({std::nullopt})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<bool>({false})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeNullableFlatVector<bool>({false})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<bool>({false})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<bool>({false})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeNullableFlatVector<bool>({true})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<bool>({false})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 7: Multi-column, both inclusive (bool + bool)
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<bool>({false}), makeFlatVector<bool>({false})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<bool>({true}), makeFlatVector<bool>({true})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<bool>({false}),
             makeFlatVector<bool>({false})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper (overflow)
        makeRowVector(
            {makeFlatVector<bool>({false}),
             makeFlatVector<bool>({false})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper (overflow)
        makeRowVector(
            {makeFlatVector<bool>({false}),
             makeFlatVector<bool>({false})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<bool>({true}),
             makeFlatVector<bool>({false})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<bool>({false}),
             makeFlatVector<bool>({false})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<bool>({true}),
             makeFlatVector<bool>({false})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 8: Multi-column with null in first column of lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<bool>({std::nullopt}),
                 makeFlatVector<bool>({false})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<bool>({true}), makeFlatVector<bool>({true})}),
            .inclusive = true},
        makeRowVector(
            {makeNullableFlatVector<bool>({std::nullopt}),
             makeFlatVector<bool>({false})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper (overflow)
        makeRowVector(
            {makeNullableFlatVector<bool>({std::nullopt}),
             makeFlatVector<bool>({false})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper (overflow)
        makeRowVector(
            {makeNullableFlatVector<bool>({std::nullopt}),
             makeFlatVector<bool>({false})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<bool>({true}),
             makeFlatVector<bool>({false})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeNullableFlatVector<bool>({std::nullopt}),
             makeFlatVector<bool>({false})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<bool>({true}),
             makeFlatVector<bool>({false})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 9: Lower bound bump failure - exclusive bound at max value
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<bool>({true})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsFirst;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 10: Lower bound bump failure - exclusive bound with null at end
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector({makeNullableFlatVector<bool>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 11: Multi-column lower bound bump failure - all columns at max
  // value
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<bool>({true}), makeFlatVector<bool>({true})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 12: Multi-column lower bound bump failure - all columns at max
  // value with null
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<bool>({std::nullopt}),
             makeNullableFlatVector<bool>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 13: Multi-column point lookup where the lower (second) column
  // hits max value (true) in ASC order. This tests carry-over behavior when
  // incrementing - when c1 is true, the increment should carry over to c0,
  // resulting in (true, false).
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<bool>({false}), makeFlatVector<bool>({true})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<bool>({false}), makeFlatVector<bool>({true})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<bool>({false}),
             makeFlatVector<bool>({true})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<bool>({true}),
             makeFlatVector<bool>({false})}), // ASC_NULLS_FIRST upper
                                              // (carry-over)
        makeRowVector(
            {makeFlatVector<bool>({false}),
             makeFlatVector<bool>({true})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<bool>({true}),
             makeFlatVector<bool>({false})}), // ASC_NULLS_LAST upper
                                              // (carry-over)
        makeRowVector(
            {makeFlatVector<bool>({false}),
             makeFlatVector<bool>({true})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<bool>({false}),
             makeFlatVector<bool>({false})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<bool>({false}),
             makeFlatVector<bool>({true})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<bool>({false}),
             makeFlatVector<bool>({false})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 14: Multi-column point lookup where the lower (second) column
  // hits min value (false) in DESC order. This tests carry-over behavior when
  // decrementing - when c1 is false, the decrement should carry over to c0,
  // resulting in (false, true).
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<bool>({true}), makeFlatVector<bool>({false})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<bool>({true}), makeFlatVector<bool>({false})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<bool>({true}),
             makeFlatVector<bool>({false})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<bool>({true}),
             makeFlatVector<bool>({true})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<bool>({true}),
             makeFlatVector<bool>({false})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<bool>({true}),
             makeFlatVector<bool>({true})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<bool>({true}),
             makeFlatVector<bool>({false})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<bool>({false}),
             makeFlatVector<bool>({true})}), // DESC_NULLS_FIRST upper
                                             // (carry-over)
        makeRowVector(
            {makeFlatVector<bool>({true}),
             makeFlatVector<bool>({false})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<bool>({false}),
             makeFlatVector<bool>({true})})); // DESC_NULLS_LAST upper
                                              // (carry-over)
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }
}

TEST_F(KeyEncoderTest, encodeIndexBoundsWithRealType) {
  // Test Case 1: Both bounds inclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<float>({10.5f})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<float>({100.5f})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<float>({10.5f})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<float>({std::nextafter(
            100.5f,
            std::numeric_limits<float>::infinity())})}), // ASC_NULLS_FIRST
                                                         // upper
        makeRowVector({makeFlatVector<float>({10.5f})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<float>({std::nextafter(
            100.5f,
            std::numeric_limits<float>::infinity())})}), // ASC_NULLS_LAST
                                                         // upper
        makeRowVector(
            {makeFlatVector<float>({10.5f})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<float>({std::nextafter(
            100.5f,
            -std::numeric_limits<float>::infinity())})}), // DESC_NULLS_FIRST
                                                          // upper
        makeRowVector(
            {makeFlatVector<float>({10.5f})}), // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<float>({std::nextafter(
            100.5f,
            -std::numeric_limits<float>::infinity())})})); // DESC_NULLS_LAST
                                                           // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 2: Both bounds exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<float>({10.5f})}),
            .inclusive = false},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<float>({100.5f})}),
            .inclusive = false},
        makeRowVector({makeFlatVector<float>({std::nextafter(
            10.5f,
            std::numeric_limits<float>::infinity())})}), // ASC_NULLS_FIRST
                                                         // lower
        makeRowVector(
            {makeFlatVector<float>({100.5f})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<float>({std::nextafter(
            10.5f,
            std::numeric_limits<float>::infinity())})}), // ASC_NULLS_LAST
                                                         // lower
        makeRowVector(
            {makeFlatVector<float>({100.5f})}), // ASC_NULLS_LAST upper
        makeRowVector({makeFlatVector<float>({std::nextafter(
            10.5f,
            -std::numeric_limits<float>::infinity())})}), // DESC_NULLS_FIRST
                                                          // lower
        makeRowVector(
            {makeFlatVector<float>({100.5f})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<float>({std::nextafter(
            10.5f,
            -std::numeric_limits<float>::infinity())})}), // DESC_NULLS_LAST
                                                          // lower
        makeRowVector(
            {makeFlatVector<float>({100.5f})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 3: Lower inclusive, upper exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<float>({10.5f})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<float>({100.5f})}),
            .inclusive = false},
        makeRowVector(
            {makeFlatVector<float>({10.5f})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<float>({100.5f})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<float>({10.5f})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<float>({100.5f})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<float>({10.5f})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<float>({100.5f})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<float>({10.5f})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<float>({100.5f})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 4: Only lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<float>({10.5f})}),
            .inclusive = true},
        std::nullopt,
        makeRowVector(
            {makeFlatVector<float>({10.5f})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<float>({10.5f})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<float>({10.5f})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<float>({10.5f})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 5: Only upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector({makeFlatVector<float>({100.5f})}),
            .inclusive = true},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<float>({std::nextafter(
            100.5f,
            std::numeric_limits<float>::infinity())})}), // ASC_NULLS_FIRST
                                                         // upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<float>({std::nextafter(
            100.5f,
            std::numeric_limits<float>::infinity())})}), // ASC_NULLS_LAST
                                                         // upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<float>({std::nextafter(
            100.5f,
            -std::numeric_limits<float>::infinity())})}), // DESC_NULLS_FIRST
                                                          // upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<float>({std::nextafter(
            100.5f,
            -std::numeric_limits<float>::infinity())})})); // DESC_NULLS_LAST
                                                           // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 6: Multi-column, both inclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<float>({10.5f}),
                 makeFlatVector<float>({20.5f})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<float>({100.5f}),
                 makeFlatVector<float>({100.5f})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<float>({10.5f}),
             makeFlatVector<float>({20.5f})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({std::nextafter(
                 100.5f,
                 std::numeric_limits<float>::infinity())})}), // ASC_NULLS_FIRST
                                                              // upper
        makeRowVector(
            {makeFlatVector<float>({10.5f}),
             makeFlatVector<float>({20.5f})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({std::nextafter(
                 100.5f,
                 std::numeric_limits<float>::infinity())})}), // ASC_NULLS_LAST
                                                              // upper
        makeRowVector(
            {makeFlatVector<float>({10.5f}),
             makeFlatVector<float>({20.5f})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({std::nextafter(
                 100.5f,
                 -std::numeric_limits<
                     float>::infinity())})}), // DESC_NULLS_FIRST
                                              // upper
        makeRowVector(
            {makeFlatVector<float>({10.5f}),
             makeFlatVector<float>({20.5f})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({std::nextafter(
                 100.5f,
                 -std::numeric_limits<
                     float>::infinity())})})); // DESC_NULLS_LAST
                                               // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 7: Single column with null in lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound =
                makeRowVector({makeNullableFlatVector<float>({std::nullopt})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<float>({100.5f})}),
            .inclusive = true},
        makeRowVector({makeNullableFlatVector<float>(
            {std::nullopt})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<float>({std::nextafter(
            100.5f,
            std::numeric_limits<float>::infinity())})}), // ASC_NULLS_FIRST
                                                         // upper
        makeRowVector({makeNullableFlatVector<float>(
            {std::nullopt})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<float>({std::nextafter(
            100.5f,
            std::numeric_limits<float>::infinity())})}), // ASC_NULLS_LAST
                                                         // upper
        makeRowVector({makeNullableFlatVector<float>(
            {std::nullopt})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<float>({std::nextafter(
            100.5f,
            -std::numeric_limits<float>::infinity())})}), // DESC_NULLS_FIRST
                                                          // upper
        makeRowVector({makeNullableFlatVector<float>(
            {std::nullopt})}), // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<float>({std::nextafter(
            100.5f,
            -std::numeric_limits<float>::infinity())})})); // DESC_NULLS_LAST
                                                           // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 8: Single column with null in upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<float>({10.5f})}),
            .inclusive = true},
        IndexBound{
            .bound =
                makeRowVector({makeNullableFlatVector<float>({std::nullopt})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<float>({10.5f})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeNullableFlatVector<float>(
            {-std::numeric_limits<float>::infinity()})}), // ASC_NULLS_FIRST
                                                          // upper
        makeRowVector({makeFlatVector<float>({10.5f})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<float>({10.5f})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeNullableFlatVector<float>(
            {std::numeric_limits<float>::infinity()})}), // DESC_NULLS_FIRST
                                                         // upper
        makeRowVector(
            {makeFlatVector<float>({10.5f})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 9: Multi-column with null in first column of lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<float>({std::nullopt}),
                 makeFlatVector<float>({20.5f})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<float>({100.5f}),
                 makeFlatVector<float>({100.5f})}),
            .inclusive = true},
        makeRowVector(
            {makeNullableFlatVector<float>({std::nullopt}),
             makeFlatVector<float>({20.5f})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({std::nextafter(
                 100.5f,
                 std::numeric_limits<float>::infinity())})}), // ASC_NULLS_FIRST
                                                              // upper
        makeRowVector(
            {makeNullableFlatVector<float>({std::nullopt}),
             makeFlatVector<float>({20.5f})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({std::nextafter(
                 100.5f,
                 std::numeric_limits<float>::infinity())})}), // ASC_NULLS_LAST
                                                              // upper
        makeRowVector(
            {makeNullableFlatVector<float>({std::nullopt}),
             makeFlatVector<float>({20.5f})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({std::nextafter(
                 100.5f,
                 -std::numeric_limits<
                     float>::infinity())})}), // DESC_NULLS_FIRST
                                              // upper
        makeRowVector(
            {makeNullableFlatVector<float>({std::nullopt}),
             makeFlatVector<float>({20.5f})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({std::nextafter(
                 100.5f,
                 -std::numeric_limits<
                     float>::infinity())})})); // DESC_NULLS_LAST
                                               // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 10: Multi-column with null in second column of lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<float>({10.5f}),
                 makeNullableFlatVector<float>({std::nullopt})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<float>({100.5f}),
                 makeFlatVector<float>({100.5f})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<float>({10.5f}),
             makeNullableFlatVector<float>(
                 {std::nullopt})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({std::nextafter(
                 100.5f,
                 std::numeric_limits<float>::infinity())})}), // ASC_NULLS_FIRST
                                                              // upper
        makeRowVector(
            {makeFlatVector<float>({10.5f}),
             makeNullableFlatVector<float>(
                 {std::nullopt})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({std::nextafter(
                 100.5f,
                 std::numeric_limits<float>::infinity())})}), // ASC_NULLS_LAST
                                                              // upper
        makeRowVector(
            {makeFlatVector<float>({10.5f}),
             makeNullableFlatVector<float>(
                 {std::nullopt})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({std::nextafter(
                 100.5f,
                 -std::numeric_limits<
                     float>::infinity())})}), // DESC_NULLS_FIRST
                                              // upper
        makeRowVector(
            {makeFlatVector<float>({10.5f}),
             makeNullableFlatVector<float>(
                 {std::nullopt})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({std::nextafter(
                 100.5f,
                 -std::numeric_limits<
                     float>::infinity())})})); // DESC_NULLS_LAST
                                               // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 11: Lower bound bump failure - exclusive bound at positive
  // infinity
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<float>({std::numeric_limits<float>::infinity()})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsFirst;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 12: Lower bound bump failure - exclusive bound with null at end
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector({makeNullableFlatVector<float>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 13: Multi-column lower bound bump failure - all columns at max
  // value
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<float>({std::numeric_limits<float>::infinity()}),
             makeFlatVector<float>({std::numeric_limits<float>::infinity()})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 14: Multi-column lower bound bump failure - all columns at max
  // value with null
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<float>({std::nullopt}),
             makeNullableFlatVector<float>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 15: Multi-column point lookup where the lower (second) column
  // hits positive infinity in ASC order. This tests carry-over behavior when
  // incrementing - when c1 is at +infinity, the increment should carry over
  // to c0, resulting in (nextafter(c0, +inf), -inf).
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<float>({100.5f}),
                 makeFlatVector<float>(
                     {std::numeric_limits<float>::infinity()})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<float>({100.5f}),
                 makeFlatVector<float>(
                     {std::numeric_limits<float>::infinity()})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>(
                 {std::numeric_limits<float>::infinity()})}), // ASC_NULLS_FIRST
                                                              // lower
        makeRowVector(
            {makeFlatVector<float>({std::nextafter(
                 100.5f, std::numeric_limits<float>::infinity())}),
             makeFlatVector<float>({-std::numeric_limits<
                 float>::infinity()})}), // ASC_NULLS_FIRST
                                         // upper
                                         // (carry-over)
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>(
                 {std::numeric_limits<float>::infinity()})}), // ASC_NULLS_LAST
                                                              // lower
        makeRowVector(
            {makeFlatVector<float>({std::nextafter(
                 100.5f, std::numeric_limits<float>::infinity())}),
             makeFlatVector<float>(
                 {-std::numeric_limits<float>::infinity()})}), // ASC_NULLS_LAST
                                                               // upper
                                                               // (carry-over)
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({std::numeric_limits<
                 float>::infinity()})}), // DESC_NULLS_FIRST
                                         // lower
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({std::nextafter(
                 std::numeric_limits<float>::infinity(),
                 -std::numeric_limits<
                     float>::infinity())})}), // DESC_NULLS_FIRST
                                              // upper
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>(
                 {std::numeric_limits<float>::infinity()})}), // DESC_NULLS_LAST
                                                              // lower
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({std::nextafter(
                 std::numeric_limits<float>::infinity(),
                 -std::numeric_limits<
                     float>::infinity())})})); // DESC_NULLS_LAST
                                               // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 16: Multi-column point lookup where the lower (second) column
  // hits negative infinity in DESC order. This tests carry-over behavior when
  // decrementing - when c1 is at -infinity, the decrement should carry over
  // to c0, resulting in (nextafter(c0, -inf), +inf).
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<float>({100.5f}),
                 makeFlatVector<float>(
                     {-std::numeric_limits<float>::infinity()})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<float>({100.5f}),
                 makeFlatVector<float>(
                     {-std::numeric_limits<float>::infinity()})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({-std::numeric_limits<
                 float>::infinity()})}), // ASC_NULLS_FIRST
                                         // lower
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({std::nextafter(
                 -std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::infinity())})}), // ASC_NULLS_FIRST
                                                              // upper
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>(
                 {-std::numeric_limits<float>::infinity()})}), // ASC_NULLS_LAST
                                                               // lower
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({std::nextafter(
                 -std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::infinity())})}), // ASC_NULLS_LAST
                                                              // upper
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({-std::numeric_limits<
                 float>::infinity()})}), // DESC_NULLS_FIRST
                                         // lower
        makeRowVector(
            {makeFlatVector<float>({std::nextafter(
                 100.5f, -std::numeric_limits<float>::infinity())}),
             makeFlatVector<float>({std::numeric_limits<
                 float>::infinity()})}), // DESC_NULLS_FIRST
                                         // upper
                                         // (carry-over)
        makeRowVector(
            {makeFlatVector<float>({100.5f}),
             makeFlatVector<float>({-std::numeric_limits<
                 float>::infinity()})}), // DESC_NULLS_LAST
                                         // lower
        makeRowVector(
            {makeFlatVector<float>({std::nextafter(
                 100.5f, -std::numeric_limits<float>::infinity())}),
             makeFlatVector<float>({std::numeric_limits<
                 float>::infinity()})})); // DESC_NULLS_LAST
                                          // upper
                                          // (carry-over)
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }
}

TEST_F(KeyEncoderTest, encodeIndexBoundsWithDoubleType) {
  // Test Case 1: Both bounds inclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<double>({10.5})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<double>({100.5})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<double>({10.5})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<double>({std::nextafter(
            100.5,
            std::numeric_limits<double>::infinity())})}), // ASC_NULLS_FIRST
                                                          // upper
        makeRowVector({makeFlatVector<double>({10.5})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<double>({std::nextafter(
            100.5,
            std::numeric_limits<double>::infinity())})}), // ASC_NULLS_LAST
                                                          // upper
        makeRowVector(
            {makeFlatVector<double>({10.5})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<double>({std::nextafter(
            100.5,
            -std::numeric_limits<double>::infinity())})}), // DESC_NULLS_FIRST
                                                           // upper
        makeRowVector(
            {makeFlatVector<double>({10.5})}), // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<double>({std::nextafter(
            100.5,
            -std::numeric_limits<double>::infinity())})})); // DESC_NULLS_LAST
                                                            // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 2: Both bounds exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<double>({10.5})}),
            .inclusive = false},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<double>({100.5})}),
            .inclusive = false},
        makeRowVector({makeFlatVector<double>({std::nextafter(
            10.5,
            std::numeric_limits<double>::infinity())})}), // ASC_NULLS_FIRST
                                                          // lower
        makeRowVector(
            {makeFlatVector<double>({100.5})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<double>({std::nextafter(
            10.5,
            std::numeric_limits<double>::infinity())})}), // ASC_NULLS_LAST
                                                          // lower
        makeRowVector(
            {makeFlatVector<double>({100.5})}), // ASC_NULLS_LAST upper
        makeRowVector({makeFlatVector<double>({std::nextafter(
            10.5,
            -std::numeric_limits<double>::infinity())})}), // DESC_NULLS_FIRST
                                                           // lower
        makeRowVector(
            {makeFlatVector<double>({100.5})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<double>({std::nextafter(
            10.5,
            -std::numeric_limits<double>::infinity())})}), // DESC_NULLS_LAST
                                                           // lower
        makeRowVector(
            {makeFlatVector<double>({100.5})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 3: Lower inclusive, upper exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<double>({10.5})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<double>({100.5})}),
            .inclusive = false},
        makeRowVector(
            {makeFlatVector<double>({10.5})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<double>({100.5})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<double>({10.5})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<double>({100.5})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<double>({10.5})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<double>({100.5})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<double>({10.5})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<double>({100.5})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 4: Only lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<double>({10.5})}),
            .inclusive = true},
        std::nullopt,
        makeRowVector(
            {makeFlatVector<double>({10.5})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<double>({10.5})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<double>({10.5})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<double>({10.5})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 5: Only upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector({makeFlatVector<double>({100.5})}),
            .inclusive = true},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<double>({std::nextafter(
            100.5,
            std::numeric_limits<double>::infinity())})}), // ASC_NULLS_FIRST
                                                          // upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<double>({std::nextafter(
            100.5,
            std::numeric_limits<double>::infinity())})}), // ASC_NULLS_LAST
                                                          // upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<double>({std::nextafter(
            100.5,
            -std::numeric_limits<double>::infinity())})}), // DESC_NULLS_FIRST
                                                           // upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<double>({std::nextafter(
            100.5,
            -std::numeric_limits<double>::infinity())})})); // DESC_NULLS_LAST
                                                            // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 6: Multi-column, both inclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<double>({10.5}),
                 makeFlatVector<double>({20.5})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<double>({100.5}),
                 makeFlatVector<double>({100.5})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<double>({10.5}),
             makeFlatVector<double>({20.5})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::nextafter(
                 100.5,
                 std::numeric_limits<
                     double>::infinity())})}), // ASC_NULLS_FIRST
                                               // upper
        makeRowVector(
            {makeFlatVector<double>({10.5}),
             makeFlatVector<double>({20.5})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::nextafter(
                 100.5,
                 std::numeric_limits<double>::infinity())})}), // ASC_NULLS_LAST
                                                               // upper
        makeRowVector(
            {makeFlatVector<double>({10.5}),
             makeFlatVector<double>({20.5})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::nextafter(
                 100.5,
                 -std::numeric_limits<
                     double>::infinity())})}), // DESC_NULLS_FIRST
                                               // upper
        makeRowVector(
            {makeFlatVector<double>({10.5}),
             makeFlatVector<double>({20.5})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::nextafter(
                 100.5,
                 -std::numeric_limits<
                     double>::infinity())})})); // DESC_NULLS_LAST
                                                // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 7: Single column with null in lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound =
                makeRowVector({makeNullableFlatVector<double>({std::nullopt})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<double>({100.5})}),
            .inclusive = true},
        makeRowVector({makeNullableFlatVector<double>(
            {std::nullopt})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<double>({std::nextafter(
            100.5,
            std::numeric_limits<double>::infinity())})}), // ASC_NULLS_FIRST
                                                          // upper
        makeRowVector({makeNullableFlatVector<double>(
            {std::nullopt})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<double>({std::nextafter(
            100.5,
            std::numeric_limits<double>::infinity())})}), // ASC_NULLS_LAST
                                                          // upper
        makeRowVector({makeNullableFlatVector<double>(
            {std::nullopt})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<double>({std::nextafter(
            100.5,
            -std::numeric_limits<double>::infinity())})}), // DESC_NULLS_FIRST
                                                           // upper
        makeRowVector({makeNullableFlatVector<double>(
            {std::nullopt})}), // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<double>({std::nextafter(
            100.5,
            -std::numeric_limits<double>::infinity())})})); // DESC_NULLS_LAST
                                                            // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 8: Single column with null in upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<double>({10.5})}),
            .inclusive = true},
        IndexBound{
            .bound =
                makeRowVector({makeNullableFlatVector<double>({std::nullopt})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<double>({10.5})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeNullableFlatVector<double>(
            {-std::numeric_limits<double>::infinity()})}), // ASC_NULLS_FIRST
                                                           // upper
        makeRowVector({makeFlatVector<double>({10.5})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<double>({10.5})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeNullableFlatVector<double>(
            {std::numeric_limits<double>::infinity()})}), // DESC_NULLS_FIRST
                                                          // upper
        makeRowVector(
            {makeFlatVector<double>({10.5})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST
                       // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 9: Multi-column with null in first column of lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<double>({std::nullopt}),
                 makeFlatVector<double>({20.5})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<double>({100.5}),
                 makeFlatVector<double>({100.5})}),
            .inclusive = true},
        makeRowVector(
            {makeNullableFlatVector<double>({std::nullopt}),
             makeFlatVector<double>({20.5})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::nextafter(
                 100.5,
                 std::numeric_limits<
                     double>::infinity())})}), // ASC_NULLS_FIRST
                                               // upper
        makeRowVector(
            {makeNullableFlatVector<double>({std::nullopt}),
             makeFlatVector<double>({20.5})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::nextafter(
                 100.5,
                 std::numeric_limits<double>::infinity())})}), // ASC_NULLS_LAST
                                                               // upper
        makeRowVector(
            {makeNullableFlatVector<double>({std::nullopt}),
             makeFlatVector<double>({20.5})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::nextafter(
                 100.5,
                 -std::numeric_limits<
                     double>::infinity())})}), // DESC_NULLS_FIRST
                                               // upper
        makeRowVector(
            {makeNullableFlatVector<double>({std::nullopt}),
             makeFlatVector<double>({20.5})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::nextafter(
                 100.5,
                 -std::numeric_limits<
                     double>::infinity())})})); // DESC_NULLS_LAST
                                                // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 10: Multi-column with null in second column of lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<double>({10.5}),
                 makeNullableFlatVector<double>({std::nullopt})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<double>({100.5}),
                 makeFlatVector<double>({100.5})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<double>({10.5}),
             makeNullableFlatVector<double>(
                 {std::nullopt})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::nextafter(
                 100.5,
                 std::numeric_limits<
                     double>::infinity())})}), // ASC_NULLS_FIRST
                                               // upper
        makeRowVector(
            {makeFlatVector<double>({10.5}),
             makeNullableFlatVector<double>(
                 {std::nullopt})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::nextafter(
                 100.5,
                 std::numeric_limits<double>::infinity())})}), // ASC_NULLS_LAST
                                                               // upper
        makeRowVector(
            {makeFlatVector<double>({10.5}),
             makeNullableFlatVector<double>(
                 {std::nullopt})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::nextafter(
                 100.5,
                 -std::numeric_limits<
                     double>::infinity())})}), // DESC_NULLS_FIRST
                                               // upper
        makeRowVector(
            {makeFlatVector<double>({10.5}),
             makeNullableFlatVector<double>(
                 {std::nullopt})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::nextafter(
                 100.5,
                 -std::numeric_limits<
                     double>::infinity())})})); // DESC_NULLS_LAST
                                                // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 11: Lower bound bump failure - exclusive bound at positive
  // infinity
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<double>(
            {std::numeric_limits<double>::infinity()})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsFirst;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 12: Lower bound bump failure - exclusive bound with null at end
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound =
            makeRowVector({makeNullableFlatVector<double>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 13: Multi-column lower bound bump failure - all columns at max
  // value
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<double>({std::numeric_limits<double>::infinity()}),
             makeFlatVector<double>(
                 {std::numeric_limits<double>::infinity()})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 14: Multi-column lower bound bump failure - all columns at max
  // value with null
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<double>({std::nullopt}),
             makeNullableFlatVector<double>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 15: Multi-column point lookup where the lower (second) column
  // hits positive infinity in ASC order. This tests carry-over behavior when
  // incrementing - when c1 is at +infinity, the increment should carry over
  // to c0, resulting in (nextafter(c0, +inf), -inf).
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<double>({100.5}),
                 makeFlatVector<double>(
                     {std::numeric_limits<double>::infinity()})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<double>({100.5}),
                 makeFlatVector<double>(
                     {std::numeric_limits<double>::infinity()})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::numeric_limits<
                 double>::infinity()})}), // ASC_NULLS_FIRST
                                          // lower
        makeRowVector(
            {makeFlatVector<double>({std::nextafter(
                 100.5, std::numeric_limits<double>::infinity())}),
             makeFlatVector<double>({-std::numeric_limits<
                 double>::infinity()})}), // ASC_NULLS_FIRST
                                          // upper
                                          // (carry-over)
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>(
                 {std::numeric_limits<double>::infinity()})}), // ASC_NULLS_LAST
                                                               // lower
        makeRowVector(
            {makeFlatVector<double>({std::nextafter(
                 100.5, std::numeric_limits<double>::infinity())}),
             makeFlatVector<double>({-std::numeric_limits<
                 double>::infinity()})}), // ASC_NULLS_LAST
                                          // upper
                                          // (carry-over)
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::numeric_limits<
                 double>::infinity()})}), // DESC_NULLS_FIRST
                                          // lower
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::nextafter(
                 std::numeric_limits<double>::infinity(),
                 -std::numeric_limits<
                     double>::infinity())})}), // DESC_NULLS_FIRST
                                               // upper
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::numeric_limits<
                 double>::infinity()})}), // DESC_NULLS_LAST
                                          // lower
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::nextafter(
                 std::numeric_limits<double>::infinity(),
                 -std::numeric_limits<
                     double>::infinity())})})); // DESC_NULLS_LAST
                                                // upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 16: Multi-column point lookup where the lower (second) column
  // hits negative infinity in DESC order. This tests carry-over behavior when
  // decrementing - when c1 is at -infinity, the decrement should carry over
  // to c0, resulting in (nextafter(c0, -inf), +inf).
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<double>({100.5}),
                 makeFlatVector<double>(
                     {-std::numeric_limits<double>::infinity()})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<double>({100.5}),
                 makeFlatVector<double>(
                     {-std::numeric_limits<double>::infinity()})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({-std::numeric_limits<
                 double>::infinity()})}), // ASC_NULLS_FIRST
                                          // lower
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::nextafter(
                 -std::numeric_limits<double>::infinity(),
                 std::numeric_limits<
                     double>::infinity())})}), // ASC_NULLS_FIRST
                                               // upper
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({-std::numeric_limits<
                 double>::infinity()})}), // ASC_NULLS_LAST
                                          // lower
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({std::nextafter(
                 -std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::infinity())})}), // ASC_NULLS_LAST
                                                               // upper
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({-std::numeric_limits<
                 double>::infinity()})}), // DESC_NULLS_FIRST
                                          // lower
        makeRowVector(
            {makeFlatVector<double>({std::nextafter(
                 100.5, -std::numeric_limits<double>::infinity())}),
             makeFlatVector<double>({std::numeric_limits<
                 double>::infinity()})}), // DESC_NULLS_FIRST
                                          // upper
                                          // (carry-over)
        makeRowVector(
            {makeFlatVector<double>({100.5}),
             makeFlatVector<double>({-std::numeric_limits<
                 double>::infinity()})}), // DESC_NULLS_LAST
                                          // lower
        makeRowVector(
            {makeFlatVector<double>({std::nextafter(
                 100.5, -std::numeric_limits<double>::infinity())}),
             makeFlatVector<double>({std::numeric_limits<
                 double>::infinity()})})); // DESC_NULLS_LAST
                                           // upper
                                           // (carry-over)
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }
}

TEST_F(KeyEncoderTest, encodeIndexBoundsWithTimestampType) {
  // Test Case 1: Both bounds inclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<velox::Timestamp>(
                {velox::Timestamp(10, 100)})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<velox::Timestamp>(
                {velox::Timestamp(100, 200)})}),
            .inclusive = true},
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(10, 100)})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 201)})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(10, 100)})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 201)})}), // ASC_NULLS_LAST upper
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(10, 100)})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 199)})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(10, 100)})}), // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 199)})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 2: Both bounds exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<velox::Timestamp>(
                {velox::Timestamp(10, 100)})}),
            .inclusive = false},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<velox::Timestamp>(
                {velox::Timestamp(100, 200)})}),
            .inclusive = false},
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(10, 101)})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 200)})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(10, 101)})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 200)})}), // ASC_NULLS_LAST upper
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(10, 99)})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 200)})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(10, 99)})}), // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 200)})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 3: Lower inclusive, upper exclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<velox::Timestamp>(
                {velox::Timestamp(10, 100)})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<velox::Timestamp>(
                {velox::Timestamp(100, 200)})}),
            .inclusive = false},
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(10, 100)})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 200)})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(10, 100)})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 200)})}), // ASC_NULLS_LAST upper
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(10, 100)})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 200)})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(10, 100)})}), // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 200)})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 4: Only lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<velox::Timestamp>(
                {velox::Timestamp(10, 100)})}),
            .inclusive = true},
        std::nullopt,
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(10, 100)})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(10, 100)})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(10, 100)})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(10, 100)})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 5: Only upper bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        std::nullopt,
        IndexBound{
            .bound = makeRowVector({makeFlatVector<velox::Timestamp>(
                {velox::Timestamp(100, 200)})}),
            .inclusive = true},
        std::nullopt, // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 201)})}), // ASC_NULLS_FIRST upper
        std::nullopt, // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 201)})}), // ASC_NULLS_LAST upper
        std::nullopt, // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 199)})}), // DESC_NULLS_FIRST upper
        std::nullopt, // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 199)})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 6: Single column with null in lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector(
                {makeNullableFlatVector<velox::Timestamp>({std::nullopt})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<velox::Timestamp>(
                {velox::Timestamp(100, 200)})}),
            .inclusive = true},
        makeRowVector({makeNullableFlatVector<velox::Timestamp>(
            {std::nullopt})}), // ASC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 201)})}), // ASC_NULLS_FIRST upper
        makeRowVector({makeNullableFlatVector<velox::Timestamp>(
            {std::nullopt})}), // ASC_NULLS_LAST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 201)})}), // ASC_NULLS_LAST upper
        makeRowVector({makeNullableFlatVector<velox::Timestamp>(
            {std::nullopt})}), // DESC_NULLS_FIRST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 199)})}), // DESC_NULLS_FIRST upper
        makeRowVector({makeNullableFlatVector<velox::Timestamp>(
            {std::nullopt})}), // DESC_NULLS_LAST lower
        makeRowVector({makeFlatVector<velox::Timestamp>(
            {velox::Timestamp(100, 199)})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 7: Multi-column, both inclusive
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<velox::Timestamp>({velox::Timestamp(10, 100)}),
                 makeFlatVector<velox::Timestamp>(
                     {velox::Timestamp(20, 200)})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
                 makeFlatVector<velox::Timestamp>(
                     {velox::Timestamp(200, 400)})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(10, 100)}),
             makeFlatVector<velox::Timestamp>(
                 {velox::Timestamp(20, 200)})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
             makeFlatVector<velox::Timestamp>(
                 {velox::Timestamp(200, 401)})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(10, 100)}),
             makeFlatVector<velox::Timestamp>(
                 {velox::Timestamp(20, 200)})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
             makeFlatVector<velox::Timestamp>(
                 {velox::Timestamp(200, 401)})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(10, 100)}),
             makeFlatVector<velox::Timestamp>(
                 {velox::Timestamp(20, 200)})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
             makeFlatVector<velox::Timestamp>(
                 {velox::Timestamp(200, 399)})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(10, 100)}),
             makeFlatVector<velox::Timestamp>(
                 {velox::Timestamp(20, 200)})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
             makeFlatVector<velox::Timestamp>(
                 {velox::Timestamp(200, 399)})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 8: Lower bound bump failure - exclusive bound at max value
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp::max()})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsFirst;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 9: Lower bound bump failure - exclusive bound with null at end
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<velox::Timestamp>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 10: Multi-column lower bound bump failure - all columns at max
  // value
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp::max()}),
             makeFlatVector<velox::Timestamp>({velox::Timestamp::max()})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 11: Multi-column lower bound bump failure - all columns at max
  // value with null
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<velox::Timestamp>({std::nullopt}),
             makeNullableFlatVector<velox::Timestamp>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 12: Multi-column point lookup where the lower (second) column
  // hits max value in ASC order. This tests carry-over behavior when
  // incrementing - when c1 is at Timestamp::max(), the increment should carry
  // over to c0, resulting in (c0+1nanos, Timestamp::min()).
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
                 makeFlatVector<velox::Timestamp>({velox::Timestamp(
                     velox::Timestamp::kMaxSeconds,
                     velox::Timestamp::kMaxNanos)})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
                 makeFlatVector<velox::Timestamp>({velox::Timestamp(
                     velox::Timestamp::kMaxSeconds,
                     velox::Timestamp::kMaxNanos)})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
             makeFlatVector<velox::Timestamp>({velox::Timestamp(
                 velox::Timestamp::kMaxSeconds,
                 velox::Timestamp::kMaxNanos)})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 301)}),
             makeFlatVector<velox::Timestamp>({velox::Timestamp(
                 velox::Timestamp::kMinSeconds, 0)})}), // ASC_NULLS_FIRST upper
                                                        // (carry-over)
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
             makeFlatVector<velox::Timestamp>({velox::Timestamp(
                 velox::Timestamp::kMaxSeconds,
                 velox::Timestamp::kMaxNanos)})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 301)}),
             makeFlatVector<velox::Timestamp>({velox::Timestamp(
                 velox::Timestamp::kMinSeconds, 0)})}), // ASC_NULLS_LAST upper
                                                        // (carry-over)
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
             makeFlatVector<velox::Timestamp>({velox::Timestamp(
                 velox::Timestamp::kMaxSeconds,
                 velox::Timestamp::kMaxNanos)})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
             makeFlatVector<velox::Timestamp>({velox::Timestamp(
                 velox::Timestamp::kMaxSeconds,
                 velox::Timestamp::kMaxNanos - 1)})}), // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
             makeFlatVector<velox::Timestamp>({velox::Timestamp(
                 velox::Timestamp::kMaxSeconds,
                 velox::Timestamp::kMaxNanos)})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
             makeFlatVector<velox::Timestamp>({velox::Timestamp(
                 velox::Timestamp::kMaxSeconds,
                 velox::Timestamp::kMaxNanos - 1)})})); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 13: Multi-column point lookup where the lower (second) column
  // hits min value in DESC order. This tests carry-over behavior when
  // decrementing - when c1 is at Timestamp::min(), the decrement should carry
  // over to c0, resulting in (c0-1nanos, Timestamp::max()).
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0", "c1"},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
                 makeFlatVector<velox::Timestamp>(
                     {velox::Timestamp(velox::Timestamp::kMinSeconds, 0)})}),
            .inclusive = true},
        IndexBound{
            .bound = makeRowVector(
                {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
                 makeFlatVector<velox::Timestamp>(
                     {velox::Timestamp(velox::Timestamp::kMinSeconds, 0)})}),
            .inclusive = true},
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
             makeFlatVector<velox::Timestamp>({velox::Timestamp(
                 velox::Timestamp::kMinSeconds, 0)})}), // ASC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
             makeFlatVector<velox::Timestamp>({velox::Timestamp(
                 velox::Timestamp::kMinSeconds, 1)})}), // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
             makeFlatVector<velox::Timestamp>({velox::Timestamp(
                 velox::Timestamp::kMinSeconds, 0)})}), // ASC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
             makeFlatVector<velox::Timestamp>({velox::Timestamp(
                 velox::Timestamp::kMinSeconds, 1)})}), // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
             makeFlatVector<velox::Timestamp>({velox::Timestamp(
                 velox::Timestamp::kMinSeconds,
                 0)})}), // DESC_NULLS_FIRST lower
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 299)}),
             makeFlatVector<velox::Timestamp>({velox::Timestamp(
                 velox::Timestamp::kMaxSeconds,
                 velox::Timestamp::kMaxNanos)})}), // DESC_NULLS_FIRST upper
                                                   // (carry-over)
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 300)}),
             makeFlatVector<velox::Timestamp>({velox::Timestamp(
                 velox::Timestamp::kMinSeconds, 0)})}), // DESC_NULLS_LAST lower
        makeRowVector(
            {makeFlatVector<velox::Timestamp>({velox::Timestamp(100, 299)}),
             makeFlatVector<velox::Timestamp>({velox::Timestamp(
                 velox::Timestamp::kMaxSeconds,
                 velox::Timestamp::kMaxNanos)})})); // DESC_NULLS_LAST upper
                                                    // (carry-over)
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }
}

TEST_F(KeyEncoderTest, encodeIndexBoundsWithStringType) {
  // Helper to create string with embedded null byte (for increment result)
  auto makeStringWithNull = [](const std::string& s) {
    return s + std::string(1, '\0');
  };

  // Test Case 1: Both bounds inclusive
  // For ascending order: increment appends '\0' (e.g., "orange" ->
  // "orange\0") For descending order: decrement only works if string ends
  // with '\0', otherwise increment fails and we get unbounded upper
  // (std::nullopt)
  {
    EncodeIndexBoundsTestCase ascNullsFirstTestCase;
    ascNullsFirstTestCase.indexColumns = {"c0"};
    ascNullsFirstTestCase.lowerBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<std::string>({"apple"})}),
        .inclusive = true};
    ascNullsFirstTestCase.upperBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<std::string>({"orange"})}),
        .inclusive = true};
    ascNullsFirstTestCase.sortOrder = velox::core::kAscNullsFirst;
    ascNullsFirstTestCase.expectedLowerBound =
        makeRowVector({makeFlatVector<std::string>({"apple"})});
    ascNullsFirstTestCase.expectedUpperBound = makeRowVector(
        {makeFlatVector<std::string>({makeStringWithNull("orange")})});
    SCOPED_TRACE(ascNullsFirstTestCase.debugString());
    testIndexBounds(ascNullsFirstTestCase);

    EncodeIndexBoundsTestCase ascNullsLastTestCase;
    ascNullsLastTestCase.indexColumns = {"c0"};
    ascNullsLastTestCase.lowerBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<std::string>({"apple"})}),
        .inclusive = true};
    ascNullsLastTestCase.upperBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<std::string>({"orange"})}),
        .inclusive = true};
    ascNullsLastTestCase.sortOrder = velox::core::kAscNullsLast;
    ascNullsLastTestCase.expectedLowerBound =
        makeRowVector({makeFlatVector<std::string>({"apple"})});
    ascNullsLastTestCase.expectedUpperBound = makeRowVector(
        {makeFlatVector<std::string>({makeStringWithNull("orange")})});
    SCOPED_TRACE(ascNullsLastTestCase.debugString());
    testIndexBounds(ascNullsLastTestCase);

    // For DESC order, decrement fails on "orange" which throws
    // VELOX_UNREACHABLE because VARCHAR filter conversion is disabled for
    // descending order.
    {
      EncodeIndexBoundsTestCase descNullsFirstTestCase;
      descNullsFirstTestCase.indexColumns = {"c0"};
      descNullsFirstTestCase.lowerBound = IndexBound{
          .bound = makeRowVector({makeFlatVector<std::string>({"apple"})}),
          .inclusive = true};
      descNullsFirstTestCase.upperBound = IndexBound{
          .bound = makeRowVector({makeFlatVector<std::string>({"orange"})}),
          .inclusive = true};
      descNullsFirstTestCase.sortOrder = velox::core::kDescNullsFirst;
      SCOPED_TRACE(descNullsFirstTestCase.debugString());
      VELOX_ASSERT_THROW(
          testIndexBounds(descNullsFirstTestCase),
          "Unexpected string underflow during descending bound increment");
    }

    {
      EncodeIndexBoundsTestCase descNullsLastTestCase;
      descNullsLastTestCase.indexColumns = {"c0"};
      descNullsLastTestCase.lowerBound = IndexBound{
          .bound = makeRowVector({makeFlatVector<std::string>({"apple"})}),
          .inclusive = true};
      descNullsLastTestCase.upperBound = IndexBound{
          .bound = makeRowVector({makeFlatVector<std::string>({"orange"})}),
          .inclusive = true};
      descNullsLastTestCase.sortOrder = velox::core::kDescNullsLast;
      SCOPED_TRACE(descNullsLastTestCase.debugString());
      VELOX_ASSERT_THROW(
          testIndexBounds(descNullsLastTestCase),
          "Unexpected string underflow during descending bound increment");
    }
  }

  // Test Case 2: Only lower bound
  {
    auto testCases = createIndexBoundEncodeTestCases(
        {"c0"},
        IndexBound{
            .bound = makeRowVector({makeFlatVector<std::string>({"apple"})}),
            .inclusive = true},
        std::nullopt,
        makeRowVector(
            {makeFlatVector<std::string>({"apple"})}), // ASC_NULLS_FIRST lower
        std::nullopt, // ASC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<std::string>({"apple"})}), // ASC_NULLS_LAST lower
        std::nullopt, // ASC_NULLS_LAST upper
        makeRowVector(
            {makeFlatVector<std::string>({"apple"})}), // DESC_NULLS_FIRST lower
        std::nullopt, // DESC_NULLS_FIRST upper
        makeRowVector(
            {makeFlatVector<std::string>({"apple"})}), // DESC_NULLS_LAST lower
        std::nullopt); // DESC_NULLS_LAST upper
    for (const auto& testCase : testCases) {
      SCOPED_TRACE(testCase.debugString());
      testIndexBounds(testCase);
    }
  }

  // Test Case 3: Only upper bound (inclusive)
  // For ASC: increment appends '\0'
  // For DESC: decrement fails on "orange" which throws VELOX_UNREACHABLE
  // because VARCHAR filter conversion is disabled for descending order.
  {
    EncodeIndexBoundsTestCase ascNullsFirstTestCase;
    ascNullsFirstTestCase.indexColumns = {"c0"};
    ascNullsFirstTestCase.lowerBound = std::nullopt;
    ascNullsFirstTestCase.upperBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<std::string>({"orange"})}),
        .inclusive = true};
    ascNullsFirstTestCase.sortOrder = velox::core::kAscNullsFirst;
    ascNullsFirstTestCase.expectedLowerBound = std::nullopt;
    ascNullsFirstTestCase.expectedUpperBound = makeRowVector(
        {makeFlatVector<std::string>({makeStringWithNull("orange")})});
    SCOPED_TRACE(ascNullsFirstTestCase.debugString());
    testIndexBounds(ascNullsFirstTestCase);

    EncodeIndexBoundsTestCase ascNullsLastTestCase;
    ascNullsLastTestCase.indexColumns = {"c0"};
    ascNullsLastTestCase.lowerBound = std::nullopt;
    ascNullsLastTestCase.upperBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<std::string>({"orange"})}),
        .inclusive = true};
    ascNullsLastTestCase.sortOrder = velox::core::kAscNullsLast;
    ascNullsLastTestCase.expectedLowerBound = std::nullopt;
    ascNullsLastTestCase.expectedUpperBound = makeRowVector(
        {makeFlatVector<std::string>({makeStringWithNull("orange")})});
    SCOPED_TRACE(ascNullsLastTestCase.debugString());
    testIndexBounds(ascNullsLastTestCase);

    // For DESC order, decrement fails on "orange" which throws
    // VELOX_UNREACHABLE because VARCHAR filter conversion is disabled for
    // descending order.
    {
      EncodeIndexBoundsTestCase descNullsFirstTestCase;
      descNullsFirstTestCase.indexColumns = {"c0"};
      descNullsFirstTestCase.lowerBound = std::nullopt;
      descNullsFirstTestCase.upperBound = IndexBound{
          .bound = makeRowVector({makeFlatVector<std::string>({"orange"})}),
          .inclusive = true};
      descNullsFirstTestCase.sortOrder = velox::core::kDescNullsFirst;
      SCOPED_TRACE(descNullsFirstTestCase.debugString());
      VELOX_ASSERT_THROW(
          testIndexBounds(descNullsFirstTestCase),
          "Unexpected string underflow during descending bound increment");
    }

    {
      EncodeIndexBoundsTestCase descNullsLastTestCase;
      descNullsLastTestCase.indexColumns = {"c0"};
      descNullsLastTestCase.lowerBound = std::nullopt;
      descNullsLastTestCase.upperBound = IndexBound{
          .bound = makeRowVector({makeFlatVector<std::string>({"orange"})}),
          .inclusive = true};
      descNullsLastTestCase.sortOrder = velox::core::kDescNullsLast;
      SCOPED_TRACE(descNullsLastTestCase.debugString());
      VELOX_ASSERT_THROW(
          testIndexBounds(descNullsLastTestCase),
          "Unexpected string underflow during descending bound increment");
    }
  }

  // Test Case 4: String ending with '\0' - decrement should work
  // This tests the Kudu-aligned behavior where decrement truncates trailing
  // '\0'
  {
    std::string stringWithNull = makeStringWithNull("abc");

    EncodeIndexBoundsTestCase descTestCase;
    descTestCase.indexColumns = {"c0"};
    descTestCase.lowerBound = std::nullopt;
    descTestCase.upperBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<std::string>({stringWithNull})}),
        .inclusive = true};
    descTestCase.sortOrder = velox::core::kDescNullsFirst;
    descTestCase.expectedLowerBound = std::nullopt;
    // Decrement of "abc\0" should give "abc"
    descTestCase.expectedUpperBound =
        makeRowVector({makeFlatVector<std::string>({"abc"})});
    SCOPED_TRACE(descTestCase.debugString());
    testIndexBounds(descTestCase);
  }

  // Test Case 5: Empty string lower bound
  {
    EncodeIndexBoundsTestCase ascNullsFirstTestCase;
    ascNullsFirstTestCase.indexColumns = {"c0"};
    ascNullsFirstTestCase.lowerBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<std::string>({""})}),
        .inclusive = true};
    ascNullsFirstTestCase.upperBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<std::string>({"abc"})}),
        .inclusive = true};
    ascNullsFirstTestCase.sortOrder = velox::core::kAscNullsFirst;
    ascNullsFirstTestCase.expectedLowerBound =
        makeRowVector({makeFlatVector<std::string>({""})});
    ascNullsFirstTestCase.expectedUpperBound = makeRowVector(
        {makeFlatVector<std::string>({makeStringWithNull("abc")})});
    SCOPED_TRACE(ascNullsFirstTestCase.debugString());
    testIndexBounds(ascNullsFirstTestCase);

    EncodeIndexBoundsTestCase ascNullsLastTestCase;
    ascNullsLastTestCase.indexColumns = {"c0"};
    ascNullsLastTestCase.lowerBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<std::string>({""})}),
        .inclusive = true};
    ascNullsLastTestCase.upperBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<std::string>({"abc"})}),
        .inclusive = true};
    ascNullsLastTestCase.sortOrder = velox::core::kAscNullsLast;
    ascNullsLastTestCase.expectedLowerBound =
        makeRowVector({makeFlatVector<std::string>({""})});
    ascNullsLastTestCase.expectedUpperBound = makeRowVector(
        {makeFlatVector<std::string>({makeStringWithNull("abc")})});
    SCOPED_TRACE(ascNullsLastTestCase.debugString());
    testIndexBounds(ascNullsLastTestCase);

    // For DESC order, decrement fails on "abc" which throws VELOX_UNREACHABLE
    // because VARCHAR filter conversion is disabled for descending order.
    {
      EncodeIndexBoundsTestCase descNullsFirstTestCase;
      descNullsFirstTestCase.indexColumns = {"c0"};
      descNullsFirstTestCase.lowerBound = IndexBound{
          .bound = makeRowVector({makeFlatVector<std::string>({""})}),
          .inclusive = true};
      descNullsFirstTestCase.upperBound = IndexBound{
          .bound = makeRowVector({makeFlatVector<std::string>({"abc"})}),
          .inclusive = true};
      descNullsFirstTestCase.sortOrder = velox::core::kDescNullsFirst;
      SCOPED_TRACE(descNullsFirstTestCase.debugString());
      VELOX_ASSERT_THROW(
          testIndexBounds(descNullsFirstTestCase),
          "Unexpected string underflow during descending bound increment");
    }

    {
      EncodeIndexBoundsTestCase descNullsLastTestCase;
      descNullsLastTestCase.indexColumns = {"c0"};
      descNullsLastTestCase.lowerBound = IndexBound{
          .bound = makeRowVector({makeFlatVector<std::string>({""})}),
          .inclusive = true};
      descNullsLastTestCase.upperBound = IndexBound{
          .bound = makeRowVector({makeFlatVector<std::string>({"abc"})}),
          .inclusive = true};
      descNullsLastTestCase.sortOrder = velox::core::kDescNullsLast;
      SCOPED_TRACE(descNullsLastTestCase.debugString());
      VELOX_ASSERT_THROW(
          testIndexBounds(descNullsLastTestCase),
          "Unexpected string underflow during descending bound increment");
    }
  }

  // Test Case 6: Multi-column, both inclusive
  // For multi-column, increment works on rightmost column first
  {
    EncodeIndexBoundsTestCase ascNullsFirstTestCase;
    ascNullsFirstTestCase.indexColumns = {"c0", "c1"};
    ascNullsFirstTestCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<std::string>({"apple"}),
             makeFlatVector<std::string>({"banana"})}),
        .inclusive = true};
    ascNullsFirstTestCase.upperBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<std::string>({"orange"}),
             makeFlatVector<std::string>({"peach"})}),
        .inclusive = true};
    ascNullsFirstTestCase.sortOrder = velox::core::kAscNullsFirst;
    ascNullsFirstTestCase.expectedLowerBound = makeRowVector(
        {makeFlatVector<std::string>({"apple"}),
         makeFlatVector<std::string>({"banana"})});
    ascNullsFirstTestCase.expectedUpperBound = makeRowVector(
        {makeFlatVector<std::string>({"orange"}),
         makeFlatVector<std::string>({makeStringWithNull("peach")})});
    SCOPED_TRACE(ascNullsFirstTestCase.debugString());
    testIndexBounds(ascNullsFirstTestCase);

    EncodeIndexBoundsTestCase ascNullsLastTestCase;
    ascNullsLastTestCase.indexColumns = {"c0", "c1"};
    ascNullsLastTestCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<std::string>({"apple"}),
             makeFlatVector<std::string>({"banana"})}),
        .inclusive = true};
    ascNullsLastTestCase.upperBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<std::string>({"orange"}),
             makeFlatVector<std::string>({"peach"})}),
        .inclusive = true};
    ascNullsLastTestCase.sortOrder = velox::core::kAscNullsLast;
    ascNullsLastTestCase.expectedLowerBound = makeRowVector(
        {makeFlatVector<std::string>({"apple"}),
         makeFlatVector<std::string>({"banana"})});
    ascNullsLastTestCase.expectedUpperBound = makeRowVector(
        {makeFlatVector<std::string>({"orange"}),
         makeFlatVector<std::string>({makeStringWithNull("peach")})});
    SCOPED_TRACE(ascNullsLastTestCase.debugString());
    testIndexBounds(ascNullsLastTestCase);

    // For DESC order, decrement fails on rightmost column "peach",
    // which throws VELOX_UNREACHABLE because VARCHAR filter conversion is
    // disabled for descending order.
    {
      EncodeIndexBoundsTestCase descNullsFirstTestCase;
      descNullsFirstTestCase.indexColumns = {"c0", "c1"};
      descNullsFirstTestCase.lowerBound = IndexBound{
          .bound = makeRowVector(
              {makeFlatVector<std::string>({"apple"}),
               makeFlatVector<std::string>({"banana"})}),
          .inclusive = true};
      descNullsFirstTestCase.upperBound = IndexBound{
          .bound = makeRowVector(
              {makeFlatVector<std::string>({"orange"}),
               makeFlatVector<std::string>({"peach"})}),
          .inclusive = true};
      descNullsFirstTestCase.sortOrder = velox::core::kDescNullsFirst;
      SCOPED_TRACE(descNullsFirstTestCase.debugString());
      VELOX_ASSERT_THROW(
          testIndexBounds(descNullsFirstTestCase),
          "Unexpected string underflow during descending bound increment");
    }

    {
      EncodeIndexBoundsTestCase descNullsLastTestCase;
      descNullsLastTestCase.indexColumns = {"c0", "c1"};
      descNullsLastTestCase.lowerBound = IndexBound{
          .bound = makeRowVector(
              {makeFlatVector<std::string>({"apple"}),
               makeFlatVector<std::string>({"banana"})}),
          .inclusive = true};
      descNullsLastTestCase.upperBound = IndexBound{
          .bound = makeRowVector(
              {makeFlatVector<std::string>({"orange"}),
               makeFlatVector<std::string>({"peach"})}),
          .inclusive = true};
      descNullsLastTestCase.sortOrder = velox::core::kDescNullsLast;
      SCOPED_TRACE(descNullsLastTestCase.debugString());
      VELOX_ASSERT_THROW(
          testIndexBounds(descNullsLastTestCase),
          "Unexpected string underflow during descending bound increment");
    }
  }

  // Test Case 7: Single column with null in lower bound
  // For null in lower bound with inclusive, the lower bound stays as null
  // For upper bound, ASC order appends '\0', DESC order fails decrement
  {
    EncodeIndexBoundsTestCase ascNullsFirstTestCase;
    ascNullsFirstTestCase.indexColumns = {"c0"};
    ascNullsFirstTestCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<std::string>({std::nullopt})}),
        .inclusive = true};
    ascNullsFirstTestCase.upperBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<std::string>({"orange"})}),
        .inclusive = true};
    ascNullsFirstTestCase.sortOrder = velox::core::kAscNullsFirst;
    ascNullsFirstTestCase.expectedLowerBound =
        makeRowVector({makeNullableFlatVector<std::string>({std::nullopt})});
    ascNullsFirstTestCase.expectedUpperBound = makeRowVector(
        {makeFlatVector<std::string>({makeStringWithNull("orange")})});
    SCOPED_TRACE(ascNullsFirstTestCase.debugString());
    testIndexBounds(ascNullsFirstTestCase);

    EncodeIndexBoundsTestCase ascNullsLastTestCase;
    ascNullsLastTestCase.indexColumns = {"c0"};
    ascNullsLastTestCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<std::string>({std::nullopt})}),
        .inclusive = true};
    ascNullsLastTestCase.upperBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<std::string>({"orange"})}),
        .inclusive = true};
    ascNullsLastTestCase.sortOrder = velox::core::kAscNullsLast;
    ascNullsLastTestCase.expectedLowerBound =
        makeRowVector({makeNullableFlatVector<std::string>({std::nullopt})});
    ascNullsLastTestCase.expectedUpperBound = makeRowVector(
        {makeFlatVector<std::string>({makeStringWithNull("orange")})});
    SCOPED_TRACE(ascNullsLastTestCase.debugString());
    testIndexBounds(ascNullsLastTestCase);

    // For DESC order, decrement fails on "orange" which throws
    // VELOX_UNREACHABLE because VARCHAR filter conversion is disabled for
    // descending order.
    {
      EncodeIndexBoundsTestCase descNullsFirstTestCase;
      descNullsFirstTestCase.indexColumns = {"c0"};
      descNullsFirstTestCase.lowerBound = IndexBound{
          .bound = makeRowVector(
              {makeNullableFlatVector<std::string>({std::nullopt})}),
          .inclusive = true};
      descNullsFirstTestCase.upperBound = IndexBound{
          .bound = makeRowVector({makeFlatVector<std::string>({"orange"})}),
          .inclusive = true};
      descNullsFirstTestCase.sortOrder = velox::core::kDescNullsFirst;
      SCOPED_TRACE(descNullsFirstTestCase.debugString());
      VELOX_ASSERT_THROW(
          testIndexBounds(descNullsFirstTestCase),
          "Unexpected string underflow during descending bound increment");
    }

    {
      EncodeIndexBoundsTestCase descNullsLastTestCase;
      descNullsLastTestCase.indexColumns = {"c0"};
      descNullsLastTestCase.lowerBound = IndexBound{
          .bound = makeRowVector(
              {makeNullableFlatVector<std::string>({std::nullopt})}),
          .inclusive = true};
      descNullsLastTestCase.upperBound = IndexBound{
          .bound = makeRowVector({makeFlatVector<std::string>({"orange"})}),
          .inclusive = true};
      descNullsLastTestCase.sortOrder = velox::core::kDescNullsLast;
      SCOPED_TRACE(descNullsLastTestCase.debugString());
      VELOX_ASSERT_THROW(
          testIndexBounds(descNullsLastTestCase),
          "Unexpected string underflow during descending bound increment");
    }
  }

  // Test Case 8: Single column with null in upper bound
  // Incrementing null in ASC_NULLS_FIRST gives empty string ""
  // For ASC_NULLS_LAST, incrementing null gives "" (carry to first column)
  // For DESC orders, decrement fails on strings
  {
    // ASC_NULLS_FIRST: null sorts first, incrementing gives ""
    EncodeIndexBoundsTestCase ascNullsFirstTestCase;
    ascNullsFirstTestCase.indexColumns = {"c0"};
    ascNullsFirstTestCase.lowerBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<std::string>({"apple"})}),
        .inclusive = true};
    ascNullsFirstTestCase.upperBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<std::string>({std::nullopt})}),
        .inclusive = true};
    ascNullsFirstTestCase.sortOrder = velox::core::kAscNullsFirst;
    ascNullsFirstTestCase.expectedLowerBound =
        makeRowVector({makeFlatVector<std::string>({"apple"})});
    ascNullsFirstTestCase.expectedUpperBound =
        makeRowVector({makeFlatVector<std::string>({""})});
    SCOPED_TRACE(ascNullsFirstTestCase.debugString());
    testIndexBounds(ascNullsFirstTestCase);

    // ASC_NULLS_LAST: null is at end, incrementing null resets to ""
    // For single column, this causes carry with no more columns,
    // so the upper bound becomes unbounded (nullopt)
    {
      EncodeIndexBoundsTestCase ascNullsLastTestCase;
      ascNullsLastTestCase.indexColumns = {"c0"};
      ascNullsLastTestCase.lowerBound = IndexBound{
          .bound = makeRowVector({makeFlatVector<std::string>({"apple"})}),
          .inclusive = true};
      ascNullsLastTestCase.upperBound = IndexBound{
          .bound = makeRowVector(
              {makeNullableFlatVector<std::string>({std::nullopt})}),
          .inclusive = true};
      ascNullsLastTestCase.sortOrder = velox::core::kAscNullsLast;
      ascNullsLastTestCase.expectedLowerBound =
          makeRowVector({makeFlatVector<std::string>({"apple"})});
      // Single column null with NULLS_LAST: carry causes overflow, no upper
      // bound
      ascNullsLastTestCase.expectedUpperBound = std::nullopt;
      SCOPED_TRACE(ascNullsLastTestCase.debugString());
      testIndexBounds(ascNullsLastTestCase);
    }

    // DESC_NULLS_FIRST: null sorts first (min), incrementing null requires
    // setMaxValueTyped which fails for strings
    {
      EncodeIndexBoundsTestCase descNullsFirstTestCase;
      descNullsFirstTestCase.indexColumns = {"c0"};
      descNullsFirstTestCase.lowerBound = IndexBound{
          .bound = makeRowVector({makeFlatVector<std::string>({"apple"})}),
          .inclusive = true};
      descNullsFirstTestCase.upperBound = IndexBound{
          .bound = makeRowVector(
              {makeNullableFlatVector<std::string>({std::nullopt})}),
          .inclusive = true};
      descNullsFirstTestCase.sortOrder = velox::core::kDescNullsFirst;
      SCOPED_TRACE(descNullsFirstTestCase.debugString());
      VELOX_ASSERT_THROW(
          testIndexBounds(descNullsFirstTestCase),
          "Cannot set max value for type VARCHAR when incrementing NULL");
    }

    // DESC_NULLS_LAST: null is at end, incrementing null requires
    // setMaxValueTyped which fails for strings
    {
      EncodeIndexBoundsTestCase descNullsLastTestCase;
      descNullsLastTestCase.indexColumns = {"c0"};
      descNullsLastTestCase.lowerBound = IndexBound{
          .bound = makeRowVector({makeFlatVector<std::string>({"apple"})}),
          .inclusive = true};
      descNullsLastTestCase.upperBound = IndexBound{
          .bound = makeRowVector(
              {makeNullableFlatVector<std::string>({std::nullopt})}),
          .inclusive = true};
      descNullsLastTestCase.sortOrder = velox::core::kDescNullsLast;
      SCOPED_TRACE(descNullsLastTestCase.debugString());
      VELOX_ASSERT_THROW(
          testIndexBounds(descNullsLastTestCase),
          "Cannot set max value for type VARCHAR when incrementing NULL");
    }
  }

  // Test Case 9: Multi-column with null in first column lower bound
  // For ASC order, upper bound appends '\0' to rightmost column
  // For DESC order, decrement fails on rightmost column "peach"
  {
    EncodeIndexBoundsTestCase ascNullsFirstTestCase;
    ascNullsFirstTestCase.indexColumns = {"c0", "c1"};
    ascNullsFirstTestCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<std::string>({std::nullopt}),
             makeFlatVector<std::string>({"banana"})}),
        .inclusive = true};
    ascNullsFirstTestCase.upperBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<std::string>({"orange"}),
             makeFlatVector<std::string>({"peach"})}),
        .inclusive = true};
    ascNullsFirstTestCase.sortOrder = velox::core::kAscNullsFirst;
    ascNullsFirstTestCase.expectedLowerBound = makeRowVector(
        {makeNullableFlatVector<std::string>({std::nullopt}),
         makeFlatVector<std::string>({"banana"})});
    ascNullsFirstTestCase.expectedUpperBound = makeRowVector(
        {makeFlatVector<std::string>({"orange"}),
         makeFlatVector<std::string>({makeStringWithNull("peach")})});
    SCOPED_TRACE(ascNullsFirstTestCase.debugString());
    testIndexBounds(ascNullsFirstTestCase);

    EncodeIndexBoundsTestCase ascNullsLastTestCase;
    ascNullsLastTestCase.indexColumns = {"c0", "c1"};
    ascNullsLastTestCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<std::string>({std::nullopt}),
             makeFlatVector<std::string>({"banana"})}),
        .inclusive = true};
    ascNullsLastTestCase.upperBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<std::string>({"orange"}),
             makeFlatVector<std::string>({"peach"})}),
        .inclusive = true};
    ascNullsLastTestCase.sortOrder = velox::core::kAscNullsLast;
    ascNullsLastTestCase.expectedLowerBound = makeRowVector(
        {makeNullableFlatVector<std::string>({std::nullopt}),
         makeFlatVector<std::string>({"banana"})});
    ascNullsLastTestCase.expectedUpperBound = makeRowVector(
        {makeFlatVector<std::string>({"orange"}),
         makeFlatVector<std::string>({makeStringWithNull("peach")})});
    SCOPED_TRACE(ascNullsLastTestCase.debugString());
    testIndexBounds(ascNullsLastTestCase);

    // DESC order: decrement fails on both columns which throws
    // VELOX_UNREACHABLE because VARCHAR filter conversion is disabled for
    // descending order.
    {
      EncodeIndexBoundsTestCase descNullsFirstTestCase;
      descNullsFirstTestCase.indexColumns = {"c0", "c1"};
      descNullsFirstTestCase.lowerBound = IndexBound{
          .bound = makeRowVector(
              {makeNullableFlatVector<std::string>({std::nullopt}),
               makeFlatVector<std::string>({"banana"})}),
          .inclusive = true};
      descNullsFirstTestCase.upperBound = IndexBound{
          .bound = makeRowVector(
              {makeFlatVector<std::string>({"orange"}),
               makeFlatVector<std::string>({"peach"})}),
          .inclusive = true};
      descNullsFirstTestCase.sortOrder = velox::core::kDescNullsFirst;
      SCOPED_TRACE(descNullsFirstTestCase.debugString());
      VELOX_ASSERT_THROW(
          testIndexBounds(descNullsFirstTestCase),
          "Unexpected string underflow during descending bound increment");
    }

    {
      EncodeIndexBoundsTestCase descNullsLastTestCase;
      descNullsLastTestCase.indexColumns = {"c0", "c1"};
      descNullsLastTestCase.lowerBound = IndexBound{
          .bound = makeRowVector(
              {makeNullableFlatVector<std::string>({std::nullopt}),
               makeFlatVector<std::string>({"banana"})}),
          .inclusive = true};
      descNullsLastTestCase.upperBound = IndexBound{
          .bound = makeRowVector(
              {makeFlatVector<std::string>({"orange"}),
               makeFlatVector<std::string>({"peach"})}),
          .inclusive = true};
      descNullsLastTestCase.sortOrder = velox::core::kDescNullsLast;
      SCOPED_TRACE(descNullsLastTestCase.debugString());
      VELOX_ASSERT_THROW(
          testIndexBounds(descNullsLastTestCase),
          "Unexpected string underflow during descending bound increment");
    }
  }

  // Test Case 10: Multi-column with null in second column lower bound
  {
    EncodeIndexBoundsTestCase ascNullsFirstTestCase;
    ascNullsFirstTestCase.indexColumns = {"c0", "c1"};
    ascNullsFirstTestCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<std::string>({"apple"}),
             makeNullableFlatVector<std::string>({std::nullopt})}),
        .inclusive = true};
    ascNullsFirstTestCase.upperBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<std::string>({"orange"}),
             makeFlatVector<std::string>({"peach"})}),
        .inclusive = true};
    ascNullsFirstTestCase.sortOrder = velox::core::kAscNullsFirst;
    ascNullsFirstTestCase.expectedLowerBound = makeRowVector(
        {makeFlatVector<std::string>({"apple"}),
         makeNullableFlatVector<std::string>({std::nullopt})});
    ascNullsFirstTestCase.expectedUpperBound = makeRowVector(
        {makeFlatVector<std::string>({"orange"}),
         makeFlatVector<std::string>({makeStringWithNull("peach")})});
    SCOPED_TRACE(ascNullsFirstTestCase.debugString());
    testIndexBounds(ascNullsFirstTestCase);

    EncodeIndexBoundsTestCase ascNullsLastTestCase;
    ascNullsLastTestCase.indexColumns = {"c0", "c1"};
    ascNullsLastTestCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<std::string>({"apple"}),
             makeNullableFlatVector<std::string>({std::nullopt})}),
        .inclusive = true};
    ascNullsLastTestCase.upperBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<std::string>({"orange"}),
             makeFlatVector<std::string>({"peach"})}),
        .inclusive = true};
    ascNullsLastTestCase.sortOrder = velox::core::kAscNullsLast;
    // For NULLS_LAST, null is at the end, no increment needed for lower bound
    ascNullsLastTestCase.expectedLowerBound = makeRowVector(
        {makeFlatVector<std::string>({"apple"}),
         makeNullableFlatVector<std::string>({std::nullopt})});
    ascNullsLastTestCase.expectedUpperBound = makeRowVector(
        {makeFlatVector<std::string>({"orange"}),
         makeFlatVector<std::string>({makeStringWithNull("peach")})});
    SCOPED_TRACE(ascNullsLastTestCase.debugString());
    testIndexBounds(ascNullsLastTestCase);

    // DESC order: decrement fails which throws VELOX_UNREACHABLE
    // because VARCHAR filter conversion is disabled for descending order.
    {
      EncodeIndexBoundsTestCase descNullsFirstTestCase;
      descNullsFirstTestCase.indexColumns = {"c0", "c1"};
      descNullsFirstTestCase.lowerBound = IndexBound{
          .bound = makeRowVector(
              {makeFlatVector<std::string>({"apple"}),
               makeNullableFlatVector<std::string>({std::nullopt})}),
          .inclusive = true};
      descNullsFirstTestCase.upperBound = IndexBound{
          .bound = makeRowVector(
              {makeFlatVector<std::string>({"orange"}),
               makeFlatVector<std::string>({"peach"})}),
          .inclusive = true};
      descNullsFirstTestCase.sortOrder = velox::core::kDescNullsFirst;
      SCOPED_TRACE(descNullsFirstTestCase.debugString());
      VELOX_ASSERT_THROW(
          testIndexBounds(descNullsFirstTestCase),
          "Unexpected string underflow during descending bound increment");
    }

    {
      EncodeIndexBoundsTestCase descNullsLastTestCase;
      descNullsLastTestCase.indexColumns = {"c0", "c1"};
      descNullsLastTestCase.lowerBound = IndexBound{
          .bound = makeRowVector(
              {makeFlatVector<std::string>({"apple"}),
               makeNullableFlatVector<std::string>({std::nullopt})}),
          .inclusive = true};
      descNullsLastTestCase.upperBound = IndexBound{
          .bound = makeRowVector(
              {makeFlatVector<std::string>({"orange"}),
               makeFlatVector<std::string>({"peach"})}),
          .inclusive = true};
      descNullsLastTestCase.sortOrder = velox::core::kDescNullsLast;
      SCOPED_TRACE(descNullsLastTestCase.debugString());
      VELOX_ASSERT_THROW(
          testIndexBounds(descNullsLastTestCase),
          "Unexpected string underflow during descending bound increment");
    }
  }

  // Test Case 11: Multi-column with null in upper bound
  // For ASC_NULLS_FIRST, null sorts first so incrementing null gives ""
  // For ASC_NULLS_LAST, null sorts last, increment first column
  // For DESC orders, decrement fails on first column string
  {
    EncodeIndexBoundsTestCase ascNullsFirstTestCase;
    ascNullsFirstTestCase.indexColumns = {"c0", "c1"};
    ascNullsFirstTestCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<std::string>({"apple"}),
             makeFlatVector<std::string>({"banana"})}),
        .inclusive = true};
    ascNullsFirstTestCase.upperBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<std::string>({"orange"}),
             makeNullableFlatVector<std::string>({std::nullopt})}),
        .inclusive = true};
    ascNullsFirstTestCase.sortOrder = velox::core::kAscNullsFirst;
    ascNullsFirstTestCase.expectedLowerBound = makeRowVector(
        {makeFlatVector<std::string>({"apple"}),
         makeFlatVector<std::string>({"banana"})});
    // Incrementing null in NULLS_FIRST gives ""
    ascNullsFirstTestCase.expectedUpperBound = makeRowVector(
        {makeFlatVector<std::string>({"orange"}),
         makeFlatVector<std::string>({""})});
    SCOPED_TRACE(ascNullsFirstTestCase.debugString());
    testIndexBounds(ascNullsFirstTestCase);

    EncodeIndexBoundsTestCase ascNullsLastTestCase;
    ascNullsLastTestCase.indexColumns = {"c0", "c1"};
    ascNullsLastTestCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<std::string>({"apple"}),
             makeFlatVector<std::string>({"banana"})}),
        .inclusive = true};
    ascNullsLastTestCase.upperBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<std::string>({"orange"}),
             makeNullableFlatVector<std::string>({std::nullopt})}),
        .inclusive = true};
    ascNullsLastTestCase.sortOrder = velox::core::kAscNullsLast;
    ascNullsLastTestCase.expectedLowerBound = makeRowVector(
        {makeFlatVector<std::string>({"apple"}),
         makeFlatVector<std::string>({"banana"})});
    // Null at NULLS_LAST resets to "" for carry, increment first column
    ascNullsLastTestCase.expectedUpperBound = makeRowVector(
        {makeFlatVector<std::string>({makeStringWithNull("orange")}),
         makeFlatVector<std::string>({""})});
    SCOPED_TRACE(ascNullsLastTestCase.debugString());
    testIndexBounds(ascNullsLastTestCase);

    // DESC order: decrement fails on null in second column, then fails on
    // first column "orange" which throws VELOX_UNREACHABLE because VARCHAR
    // filter conversion is disabled for descending order.
    {
      EncodeIndexBoundsTestCase descNullsFirstTestCase;
      descNullsFirstTestCase.indexColumns = {"c0", "c1"};
      descNullsFirstTestCase.lowerBound = IndexBound{
          .bound = makeRowVector(
              {makeFlatVector<std::string>({"apple"}),
               makeFlatVector<std::string>({"banana"})}),
          .inclusive = true};
      descNullsFirstTestCase.upperBound = IndexBound{
          .bound = makeRowVector(
              {makeFlatVector<std::string>({"orange"}),
               makeNullableFlatVector<std::string>({std::nullopt})}),
          .inclusive = true};
      descNullsFirstTestCase.sortOrder = velox::core::kDescNullsFirst;
      SCOPED_TRACE(descNullsFirstTestCase.debugString());
      VELOX_ASSERT_THROW(
          testIndexBounds(descNullsFirstTestCase),
          "Cannot set max value for type VARCHAR when incrementing NULL");
    }

    {
      EncodeIndexBoundsTestCase descNullsLastTestCase;
      descNullsLastTestCase.indexColumns = {"c0", "c1"};
      descNullsLastTestCase.lowerBound = IndexBound{
          .bound = makeRowVector(
              {makeFlatVector<std::string>({"apple"}),
               makeFlatVector<std::string>({"banana"})}),
          .inclusive = true};
      descNullsLastTestCase.upperBound = IndexBound{
          .bound = makeRowVector(
              {makeFlatVector<std::string>({"orange"}),
               makeNullableFlatVector<std::string>({std::nullopt})}),
          .inclusive = true};
      descNullsLastTestCase.sortOrder = velox::core::kDescNullsLast;
      SCOPED_TRACE(descNullsLastTestCase.debugString());
      VELOX_ASSERT_THROW(
          testIndexBounds(descNullsLastTestCase),
          "Cannot set max value for type VARCHAR when incrementing NULL");
    }
  }

  // Test lower bound bump failures
  // Test Case 11: Single column lower bound bump failure - exclusive bound
  // with empty string. For DESC order, decrement fails which throws
  // VELOX_UNREACHABLE because VARCHAR filter conversion is disabled for
  // descending order.
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<std::string>({""})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kDescNullsFirst;
    SCOPED_TRACE(testCase.debugString());
    VELOX_ASSERT_THROW(
        testIndexBounds(testCase),
        "Unexpected string underflow during descending bound increment");
  }

  // Test Case 12: Single column lower bound bump failure - exclusive bound
  // with null at end
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<std::string>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 13: Multi-column lower bound bump failure - all columns with
  // empty strings. For DESC order, decrement fails which throws
  // VELOX_UNREACHABLE because VARCHAR filter conversion is disabled for
  // descending order.
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<std::string>({""}),
             makeFlatVector<std::string>({""})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kDescNullsFirst;
    SCOPED_TRACE(testCase.debugString());
    VELOX_ASSERT_THROW(
        testIndexBounds(testCase),
        "Unexpected string underflow during descending bound increment");
  }

  // Test Case 14: Multi-column lower bound bump failure - all columns at max
  // value with null
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<std::string>({std::nullopt}),
             makeNullableFlatVector<std::string>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }
}

// Test column separator with string + integer columns
TEST_F(KeyEncoderTest, columnSeparatorTest) {
  const std::vector<std::string> keyColumns = {"c0", "c1"};

  // Test with multiple input datasets
  const std::vector<velox::RowVectorPtr> inputs = {
      makeRowVector({
          makeFlatVector<std::string>({"0000", "00000"}),
          makeFlatVector<int32_t>({0, 0}),
      }),
      makeRowVector(
          {makeFlatVector<std::string>({"0000", "00000"}),
           makeFlatVector<int32_t>({1, 1})})};

  // Test all 4 sort order combinations
  const std::vector<velox::core::SortOrder> sortOrders = {
      velox::core::kAscNullsFirst,
      velox::core::kAscNullsLast,
      velox::core::kDescNullsFirst,
      velox::core::kDescNullsLast,
  };

  for (const auto& input : inputs) {
    SCOPED_TRACE(fmt::format("input:{}", input->toString(0, input->size())));
    for (const auto& sortOrder : sortOrders) {
      SCOPED_TRACE(
          fmt::format(
              "sortOrder: {} {}",
              sortOrder.isAscending() ? "ASC" : "DESC",
              sortOrder.isNullsFirst() ? "NULLS_FIRST" : "NULLS_LAST"));

      auto encoder = KeyEncoder::create(
          keyColumns,
          asRowType(input->type()),
          {sortOrder, sortOrder},
          pool_.get());

      // Use the public encode() method
      std::vector<char> buffer;
      std::vector<std::string_view> encodedKeys;
      encoder->encode(input, encodedKeys, [&buffer](size_t size) -> void* {
        buffer.resize(size);
        return buffer.data();
      });

      // Verify we got 2 distinct keys
      EXPECT_EQ(encodedKeys.size(), 2);
      EXPECT_NE(encodedKeys[0], encodedKeys[1]);

      if (sortOrder.isAscending()) {
        EXPECT_LT(encodedKeys[0], encodedKeys[1]);
      } else {
        EXPECT_GT(encodedKeys[0], encodedKeys[1]);
      }
    }
  }
}

// Tests for incrementNullColumnValue fix - verifies that incrementing NULL
// values throws when the type doesn't support min/max representation.
TEST_F(KeyEncoderTest, encodeIndexBoundsNullIncrementThrows) {
  // Test Case 1: VARCHAR + DESC + NULLS_FIRST - incrementing NULL needs max
  // value, which throws for strings since there's no max string representation.
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<std::string>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kDescNullsFirst;
    SCOPED_TRACE(testCase.debugString());
    VELOX_ASSERT_THROW(
        testIndexBounds(testCase),
        "Cannot set max value for type VARCHAR when incrementing NULL");
  }

  // Test Case 2: VARCHAR + DESC + NULLS_LAST - resetting NULL for carry needs
  // max value, which throws for strings.
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<std::string>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kDescNullsLast;
    SCOPED_TRACE(testCase.debugString());
    VELOX_ASSERT_THROW(
        testIndexBounds(testCase),
        "Cannot set max value for type VARCHAR when incrementing NULL");
  }

  // Test Case 3: Multi-column with VARCHAR + DESC - second column NULL needs
  // max value for carry, which throws.
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<int64_t>({5}),
             makeNullableFlatVector<std::string>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kDescNullsFirst;
    SCOPED_TRACE(testCase.debugString());
    VELOX_ASSERT_THROW(
        testIndexBounds(testCase),
        "Cannot set max value for type VARCHAR when incrementing NULL");
  }

  // Test Case 4: Integer + ASC + NULLS_FIRST - incrementing NULL gives
  // MIN_VALUE which is supported.
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound =
            makeRowVector({makeNullableFlatVector<int64_t>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsFirst;
    testCase.expectedLowerBound = makeRowVector(
        {makeFlatVector<int64_t>({std::numeric_limits<int64_t>::min()})});
    testCase.expectedUpperBound = std::nullopt;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 5: Integer + DESC + NULLS_FIRST - incrementing NULL gives
  // MAX_VALUE which is supported.
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound =
            makeRowVector({makeNullableFlatVector<int64_t>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kDescNullsFirst;
    testCase.expectedLowerBound = makeRowVector(
        {makeFlatVector<int64_t>({std::numeric_limits<int64_t>::max()})});
    testCase.expectedUpperBound = std::nullopt;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 6: Integer + ASC + NULLS_LAST - NULL is at end, can't increment,
  // returns nullopt (single column overflow).
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound =
            makeRowVector({makeNullableFlatVector<int64_t>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    testCase.expectedFailure = true;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 7: VARCHAR + ASC + NULLS_FIRST - incrementing NULL gives empty
  // string (MIN_VALUE), which is supported.
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeNullableFlatVector<std::string>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsFirst;
    testCase.expectedLowerBound =
        makeRowVector({makeFlatVector<std::string>({""})});
    testCase.expectedUpperBound = std::nullopt;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 8: Multi-column carry behavior - when rightmost column
  // overflows, it should reset and carry to the next column.
  // (5, NULL) with ASC NULLS_LAST -> (6, MIN_VALUE) for integers
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<int64_t>({5}),
             makeNullableFlatVector<int64_t>({std::nullopt})}),
        .inclusive = false};
    testCase.upperBound = std::nullopt;
    testCase.sortOrder = velox::core::kAscNullsLast;
    // NULL is at end in NULLS_LAST, so it overflows and carries to c0
    // c0 increments from 5 to 6, c1 resets to MIN_VALUE
    testCase.expectedLowerBound = makeRowVector(
        {makeFlatVector<int64_t>({6}),
         makeFlatVector<int64_t>({std::numeric_limits<int64_t>::min()})});
    testCase.expectedUpperBound = std::nullopt;
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }

  // Test Case 9: Multi-column carry behavior with DESC NULLS_LAST
  // (5, NULL) with DESC NULLS_LAST -> (4, MAX_VALUE) for integers
  // In DESC order, "increment" means decrement, and NULL at end overflows
  {
    EncodeIndexBoundsTestCase testCase;
    testCase.indexColumns = {"c0", "c1"};
    testCase.lowerBound = std::nullopt;
    testCase.upperBound = IndexBound{
        .bound = makeRowVector(
            {makeFlatVector<int64_t>({5}),
             makeNullableFlatVector<int64_t>({std::nullopt})}),
        .inclusive = true};
    testCase.sortOrder = velox::core::kDescNullsLast;
    // For inclusive upper bound, we increment to make it exclusive
    // NULL is at end in NULLS_LAST (for DESC), so it overflows and carries
    // c0 decrements from 5 to 4, c1 resets to MAX_VALUE (which is min in DESC
    // order)
    testCase.expectedLowerBound = std::nullopt;
    testCase.expectedUpperBound = makeRowVector(
        {makeFlatVector<int64_t>({4}),
         makeFlatVector<int64_t>({std::numeric_limits<int64_t>::max()})});
    SCOPED_TRACE(testCase.debugString());
    testIndexBounds(testCase);
  }
}

// Tests for multiple rows in index bounds
TEST_F(KeyEncoderTest, encodeIndexBoundsMultipleRows) {
  // Test Case 1: Multiple rows - all succeed
  {
    IndexBounds indexBounds;
    indexBounds.indexColumns = {"c0"};
    indexBounds.lowerBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<int64_t>({1, 10, 100})}),
        .inclusive = true};
    indexBounds.upperBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<int64_t>({5, 50, 500})}),
        .inclusive = true};

    ASSERT_TRUE(indexBounds.validate());

    const auto inputType = asRowType(indexBounds.type());
    const std::vector<velox::core::SortOrder> sortOrders(
        indexBounds.indexColumns.size(), velox::core::kAscNullsFirst);
    auto keyEncoder = KeyEncoder::create(
        indexBounds.indexColumns, inputType, sortOrders, pool_.get());

    const auto encodedBounds = keyEncoder->encodeIndexBounds(indexBounds);

    // Should succeed with 3 results
    ASSERT_EQ(encodedBounds.size(), 3);

    // Verify each row has both lower and upper keys
    for (size_t i = 0; i < 3; ++i) {
      EXPECT_TRUE(encodedBounds[i].lowerKey.has_value())
          << "Row " << i << " should have lowerKey";
      EXPECT_TRUE(encodedBounds[i].upperKey.has_value())
          << "Row " << i << " should have upperKey";
    }

    // Verify keys are different across rows
    EXPECT_NE(encodedBounds[0].lowerKey, encodedBounds[1].lowerKey);
    EXPECT_NE(encodedBounds[1].lowerKey, encodedBounds[2].lowerKey);
  }

  // Test Case 2: Multiple rows - one lower bound fails (throws)
  {
    IndexBounds indexBounds;
    indexBounds.indexColumns = {"c0"};
    // Second row has max value with exclusive bound - should fail to bump up
    indexBounds.lowerBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<int64_t>(
            {1, std::numeric_limits<int64_t>::max(), 100})}),
        .inclusive = false};
    indexBounds.upperBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<int64_t>({5, 50, 500})}),
        .inclusive = true};

    ASSERT_TRUE(indexBounds.validate());

    const auto inputType = asRowType(indexBounds.type());
    const std::vector<velox::core::SortOrder> sortOrders(
        indexBounds.indexColumns.size(), velox::core::kAscNullsFirst);
    auto keyEncoder = KeyEncoder::create(
        indexBounds.indexColumns, inputType, sortOrders, pool_.get());

    // Should throw because one lower bound fails to bump up
    VELOX_ASSERT_THROW(
        keyEncoder->encodeIndexBounds(indexBounds),
        "Failed to bump up lower bound");
  }

  // Test Case 3: Multiple rows - one upper bound fails (upper key becomes
  // nullopt)
  {
    IndexBounds indexBounds;
    indexBounds.indexColumns = {"c0"};
    indexBounds.lowerBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<int64_t>({1, 10, 100})}),
        .inclusive = true};
    // Second row has max value with inclusive bound - should fail to bump up
    indexBounds.upperBound = IndexBound{
        .bound = makeRowVector({makeFlatVector<int64_t>(
            {5, std::numeric_limits<int64_t>::max(), 500})}),
        .inclusive = true};

    ASSERT_TRUE(indexBounds.validate());

    const auto inputType = asRowType(indexBounds.type());
    const std::vector<velox::core::SortOrder> sortOrders(
        indexBounds.indexColumns.size(), velox::core::kAscNullsFirst);
    auto keyEncoder = KeyEncoder::create(
        indexBounds.indexColumns, inputType, sortOrders, pool_.get());

    const auto encodedBounds = keyEncoder->encodeIndexBounds(indexBounds);

    // Should succeed but all upper keys become nullopt when any fails
    ASSERT_EQ(encodedBounds.size(), 3);

    // All rows should have lower keys
    for (size_t i = 0; i < 3; ++i) {
      EXPECT_TRUE(encodedBounds[i].lowerKey.has_value())
          << "Row " << i << " should have lowerKey";
    }

    // All upper keys should be nullopt (unbounded) because bump up failed
    for (size_t i = 0; i < 3; ++i) {
      EXPECT_FALSE(encodedBounds[i].upperKey.has_value())
          << "Row " << i << " should not have upperKey (unbounded)";
    }
  }
}
} // namespace facebook::velox::serializer::test

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init{&argc, &argv, false};
  return RUN_ALL_TESTS();
}
